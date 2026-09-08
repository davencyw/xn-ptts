//! Benchmark harness for TTS generation.
//!
//! Loads a local model once, then generates the same utterance `--iters` times and reports
//! time-to-first-audio, per-frame time, total generate time and RTF. Model load and voice
//! conditioning are timed separately and excluded from the per-iteration statistics, since a
//! server pays them once and then serves many requests.
//!
//! ```bash
//! cargo run --release --features sp,accelerate --example bench -- \
//!   --model model/model.q8.gguf --config model/config.json --quant q8 \
//!   --voice voices/freya.safetensors --threads 8 --iters 20
//! ```
//!
//! Unlike `pocket_tts` this never downloads anything and only accepts precomputed voice
//! embeddings: it measures one specific model. Mimi decoding runs on the generating thread
//! rather than overlapped, so a frame's time is its sampling plus its decoding; `pocket_tts`
//! overlaps the two and will report a better RTF for the same weights.
//!
//! `--mimi-batch` decodes several frames of latent per Mimi call, which is where most of the
//! headroom on a small CPU is: see its own documentation below.

#[path = "model_helpers.rs"]
mod model_helpers;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use model_helpers::{SpTokenizer, max_frames_for};
use ptts::tts_model::{TTSConfig, TTSModel, TTSState};
use xn::{BackendQ, Tensor};

/// Frames of Mimi decoder context, matching `pocket_tts`.
const MIMI_CONTEXT_SIZE: usize = 250;

#[derive(Parser, Debug)]
#[command(name = "bench")]
#[command(about = "Benchmark TTS generation: TTFA, per-frame time, total runtime")]
struct Args {
    /// Model weights, either a safetensors file or a GGUF file (see the `quantize` example).
    #[arg(long)]
    model: std::path::PathBuf,

    /// Model config JSON.
    #[arg(long)]
    config: std::path::PathBuf,

    /// SentencePiece tokenizer. Defaults to `tokenizer.model` next to the config.
    #[arg(long)]
    tokenizer: Option<std::path::PathBuf>,

    /// Precomputed voice embedding safetensors.
    #[arg(long)]
    voice: std::path::PathBuf,

    /// Weight quantization, e.g. `q8`. Required for GGUF weights; safetensors load as f32.
    #[arg(long)]
    quant: Option<String>,

    /// Use the cpu device even if a gpu backend is available.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Number of CPU threads for tensor ops.
    #[arg(long)]
    threads: Option<usize>,

    #[arg(long, short, default_value = "Hello, this is a test of the pocket TTS system.")]
    input: String,

    #[arg(long, default_value_t = 0.4)]
    temperature: f32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Measured iterations.
    #[arg(long, default_value_t = 10, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    iters: usize,

    /// Unmeasured iterations run first, to warm caches and the thread pool.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Print a line per iteration as well as the summary.
    #[arg(long, default_value_t = false)]
    per_iter: bool,

    /// Decode this many frames of latent per Mimi call instead of one.
    ///
    /// Mimi turns each latent into 16 transformer/seanet timesteps, so decoding one frame at
    /// a time hands every gemm in the decoder m=16 -- a single microkernel row panel, with
    /// almost no reuse of the packed weight. Batching raises m proportionally and is exact:
    /// the decoder transformer is causal and the seanet convolutions are streaming, so N
    /// latents in one call produce the same samples as N calls. It costs latency, since the
    /// first chunk of audio waits for N frames to be sampled.
    #[arg(long, default_value_t = 1)]
    mimi_batch: usize,

    /// With `--pipeline`, threads for the flow LM stage and for the Mimi stage respectively,
    /// given as `SAMPLE:DECODE`.
    ///
    /// Each stage gets its own xn pool rather than sharing the process-wide one. Sharing is
    /// what makes `--pipeline` nearly free of benefit: xn's pool has a single job slot, so
    /// whichever stage publishes first fans out and the other silently runs serially on its
    /// own thread, alternating frame by frame. Sized pools let both fan out at once.
    ///
    /// The two numbers should add up to about the core count. The flow LM barely scales past
    /// one core -- it is a batch-1 autoregressive stream of small matmuls -- while the Mimi
    /// decoder scales well, so the interesting splits give Mimi most of the machine. Unset
    /// leaves both stages on the process-wide pool, i.e. the old contending behaviour.
    #[arg(long, value_name = "SAMPLE:DECODE")]
    split: Option<String>,

    /// Overlap Mimi decoding with the next frame's sampling on a second thread, as
    /// `pocket_tts` does. Frame N+1 needs only frame N's latent, never its PCM, so the decode
    /// is off the critical path. `per-frame` then reports the interval between PCM chunks --
    /// the cadence a streaming consumer sees -- instead of sampling plus decoding.
    #[arg(long, default_value_t = false)]
    pipeline: bool,

    /// Normalize the input for this language before tokenizing: `en`, `fr`, `de`, `es` or `pt`.
    /// Off by default so measurements stay comparable with runs that predate the flag.
    #[arg(long)]
    lang: Option<String>,
}

struct StdRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl StdRng {
    fn new(temperature: f32, seed: u64) -> Result<Self> {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt())?;
        Ok(Self { inner: rand::rngs::StdRng::seed_from_u64(seed), distr })
    }
}

impl ptts::flow_lm::Rng for StdRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

/// `SAMPLE:DECODE` from `--split`.
fn parse_split(s: &str) -> Result<(usize, usize)> {
    let (sample, decode) = s.split_once(':').context("--split wants SAMPLE:DECODE, e.g. 1:3")?;
    let parse = |v: &str, what: &str| -> Result<usize> {
        let n: usize =
            v.trim().parse().with_context(|| format!("--split {what} is not a number"))?;
        if n == 0 {
            anyhow::bail!("--split {what} must be at least 1")
        }
        Ok(n)
    };
    Ok((parse(sample, "sample")?, parse(decode, "decode")?))
}

/// One iteration's timings.
struct Run {
    /// Start of the iteration to the first audio samples, so text conditioning is included but
    /// the voice conditioning shared by every iteration is not.
    ttfa: Duration,
    /// Per frame, sampling plus Mimi decoding.
    frames: Vec<Duration>,
    /// Per frame, the `generate_step` half of `frames`.
    sample_t: Vec<Duration>,
    /// Per frame, the `decode_latent` half of `frames`.
    decode_t: Vec<Duration>,
    total: Duration,
    samples: usize,
}

/// Generates the utterance once, reusing the voice-conditioned state.
fn one<Q: BackendQ>(
    model: &TTSModel<Q>,
    base_state: &TTSState<Q>,
    chunks: &[(Vec<u32>, usize)],
    args: &Args,
) -> Result<Run> {
    let dev = model.device();
    let ldim = model.flow_lm.ldim;
    let mut rng = StdRng::new(args.temperature, args.seed)?;
    let mut frames = Vec::new();
    let mut sample_t = Vec::new();
    let mut decode_t = Vec::new();
    let mut ttfa = None;
    let mut samples = 0usize;
    let start = Instant::now();

    for (tokens, frames_after_eos) in chunks.iter() {
        let mut state = base_state.clone();
        model.prompt_text(&mut state, tokens)?;
        let mut mimi_state = model.init_mimi_state(1, MIMI_CONTEXT_SIZE)?;

        // BOS marker: an all-NaN latent.
        let nan: Tensor<f32, Q::B> = Tensor::from_vec(vec![f32::NAN; ldim], (1, 1, ldim), dev)?;
        let mut prev_latent = nan.to::<Q::T>()?;
        let mut eos_countdown: Option<usize> = None;
        // Latents sampled but not yet handed to Mimi, with the sampling time of each.
        let mut pending: Vec<Tensor<Q::T, Q::B>> = Vec::with_capacity(args.mimi_batch);
        let mut pending_sample: Vec<Duration> = Vec::with_capacity(args.mimi_batch);

        for i in 0..max_frames_for(tokens.len()) {
            let frame_start = Instant::now();
            let (next_latent, is_eos) = model.generate_step(&mut state, &prev_latent, &mut rng)?;
            pending_sample.push(frame_start.elapsed());
            pending.push(next_latent.clone());
            prev_latent = next_latent;

            if is_eos && eos_countdown.is_none() {
                eos_countdown = Some(*frames_after_eos);
            }
            let last = match eos_countdown.as_mut() {
                Some(0) => true,
                Some(countdown) => {
                    *countdown -= 1;
                    false
                }
                None => i + 1 == max_frames_for(tokens.len()),
            };

            // Decoding on this thread rather than overlapped, so the measurement attributes
            // sampling and decoding to the frames that caused them.
            if pending.len() == args.mimi_batch || last {
                let batch = pending.len();
                let refs: Vec<&Tensor<Q::T, Q::B>> = pending.iter().collect();
                let latents = if batch == 1 { pending[0].clone() } else { Tensor::cat(&refs, 1)? };
                let t = Instant::now();
                let pcm = model.decode_latent(&latents, &mut mimi_state)?.to_vec()?;
                // Charged evenly to the frames in the batch, so the per-frame series stays
                // comparable across `--mimi-batch` settings.
                let each = t.elapsed() / batch as u32;
                for s in pending_sample.drain(..) {
                    frames.push(s + each);
                    sample_t.push(s);
                    decode_t.push(each);
                }
                pending.clear();
                if !pcm.is_empty() {
                    ttfa.get_or_insert_with(|| start.elapsed());
                    samples += pcm.len();
                }
            }
            if last {
                break;
            }
        }
    }

    let total = start.elapsed();
    let ttfa = ttfa.context("no audio produced")?;
    Ok(Run { ttfa, frames, sample_t, decode_t, total, samples })
}

/// What the decode thread reports back.
struct Decoded {
    /// Per frame, the `decode_latent` call itself.
    decode_t: Vec<Duration>,
    /// When each non-empty PCM chunk became available, and how many frames it covered, so a
    /// `--mimi-batch` chunk can be charged to its frames rather than counted once.
    arrivals: Vec<(Instant, usize)>,
    samples: usize,
}

/// Generates the utterance once with Mimi decoding overlapped on a second thread.
///
/// The sampling thread sends each latent onward and immediately starts the next frame; a
/// scoped thread owns the `MimiDecoderState` and decodes as latents arrive. Scoped rather than
/// `spawn` so the model can be borrowed instead of shared through an `Arc`, which keeps the
/// `WithQ` impl free of a `'static` bound.
fn one_pipelined<Q: BackendQ>(
    model: &TTSModel<Q>,
    base_state: &TTSState<Q>,
    chunks: &[(Vec<u32>, usize)],
    args: &Args,
    split: Option<(usize, usize)>,
) -> Result<Run> {
    let dev = model.device();
    let ldim = model.flow_lm.ldim;
    let mut rng = StdRng::new(args.temperature, args.seed)?;
    let mut frames = Vec::new();
    let mut sample_t = Vec::new();
    let mut decode_t = Vec::new();
    let mut ttfa = None;
    let mut samples = 0usize;
    // Sampling runs on this thread. Left bound afterwards: the same thread samples every
    // iteration, and `bind` is idempotent for a given name.
    if let Some((sample, _)) = split {
        xn::threadpool::bind("flow-lm", sample);
    }
    let start = Instant::now();

    for (tokens, frames_after_eos) in chunks.iter() {
        let mut state = base_state.clone();
        model.prompt_text(&mut state, tokens)?;

        let (tx, rx) = std::sync::mpsc::channel::<Tensor<Q::T, Q::B>>();
        let decoded = std::thread::scope(|scope| -> Result<Decoded> {
            let decoder = scope.spawn(move || -> Result<Decoded> {
                // Named, so the pool is built once and reused: this thread is recreated per
                // chunk per iteration, and a fresh pool each time would leak its workers.
                if let Some((_, decode)) = split {
                    xn::threadpool::bind("mimi-decode", decode);
                }
                let mut mimi_state = model.init_mimi_state(1, MIMI_CONTEXT_SIZE)?;
                let mut out = Decoded { decode_t: Vec::new(), arrivals: Vec::new(), samples: 0 };
                while let Ok(latent) = rx.recv() {
                    let batch = latent.dim(1usize)?;
                    let t = Instant::now();
                    let pcm = model.decode_latent(&latent, &mut mimi_state)?.to_vec()?;
                    // Charged evenly to the frames in the batch, so the series stays
                    // per-frame and comparable across `--mimi-batch` settings.
                    let each = t.elapsed() / batch as u32;
                    for _ in 0..batch {
                        out.decode_t.push(each);
                    }
                    if !pcm.is_empty() {
                        out.arrivals.push((Instant::now(), batch));
                        out.samples += pcm.len();
                    }
                }
                Ok(out)
            });

            // BOS marker: an all-NaN latent.
            let nan: Tensor<f32, Q::B> = Tensor::from_vec(vec![f32::NAN; ldim], (1, 1, ldim), dev)?;
            let mut prev_latent = nan.to::<Q::T>()?;
            let mut eos_countdown: Option<usize> = None;
            let mut pending: Vec<Tensor<Q::T, Q::B>> = Vec::with_capacity(args.mimi_batch);

            for i in 0..max_frames_for(tokens.len()) {
                let frame_start = Instant::now();
                let (next_latent, is_eos) =
                    model.generate_step(&mut state, &prev_latent, &mut rng)?;
                sample_t.push(frame_start.elapsed());
                pending.push(next_latent.clone());
                prev_latent = next_latent;

                if is_eos && eos_countdown.is_none() {
                    eos_countdown = Some(*frames_after_eos);
                }
                let last = match eos_countdown.as_mut() {
                    Some(0) => true,
                    Some(countdown) => {
                        *countdown -= 1;
                        false
                    }
                    None => i + 1 == max_frames_for(tokens.len()),
                };

                if pending.len() == args.mimi_batch || last {
                    let refs: Vec<&Tensor<Q::T, Q::B>> = pending.iter().collect();
                    let latents = if pending.len() == 1 {
                        pending[0].clone()
                    } else {
                        Tensor::cat(&refs, 1)?
                    };
                    pending.clear();
                    // A send failure means the decoder died; its error surfaces on join.
                    if tx.send(latents).is_err() {
                        break;
                    }
                }
                if last {
                    break;
                }
            }
            // Close the channel so the decoder finishes, then wait for the tail of the audio.
            drop(tx);
            decoder.join().map_err(|_| anyhow::anyhow!("decode thread panicked"))?
        })?;

        // Intervals between PCM chunks, with the first measured from the start of the
        // iteration, so the series sums to the streaming wall time.
        let mut prev = start;
        for (a, batch) in decoded.arrivals.iter() {
            let each = a.duration_since(prev) / *batch as u32;
            for _ in 0..*batch {
                frames.push(each);
            }
            prev = *a;
        }
        if let Some((first, _)) = decoded.arrivals.first() {
            ttfa.get_or_insert_with(|| first.duration_since(start));
        }
        decode_t.extend(decoded.decode_t);
        samples += decoded.samples;
    }

    let total = start.elapsed();
    let ttfa = ttfa.context("no audio produced")?;
    Ok(Run { ttfa, frames, sample_t, decode_t, total, samples })
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct Stats {
    n: usize,
    min: f64,
    mean: f64,
    max: f64,
    p50: f64,
    p95: f64,
}

impl Stats {
    /// `xs` must be non-empty; `--iters` is validated to be at least 1.
    fn of(xs: &[f64]) -> Self {
        let mut s = xs.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pick = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
        Stats {
            n: s.len(),
            min: s[0],
            mean: s.iter().sum::<f64>() / s.len() as f64,
            max: s[s.len() - 1],
            p50: pick(0.50),
            p95: pick(0.95),
        }
    }
}

fn row(label: &str, unit: &str, prec: usize, st: &Stats) {
    println!(
        "{label:<22} {:>5}  {:>9.*} {:>9.*} {:>9.*} {:>9.*} {:>9.*}  {unit}",
        st.n, prec, st.min, prec, st.mean, prec, st.p50, prec, st.p95, prec, st.max
    );
}

struct Bench<'a>(&'a Args);

impl xn::WithQ for Bench<'_> {
    type Output = ();

    fn run<Q: BackendQ>(self, dev: Q::B) -> xn::Result<()> {
        self.bench::<Q>(dev).map_err(|e| xn::Error::msg(format!("{e:?}")))
    }
}

impl Bench<'_> {
    fn bench<Q: BackendQ>(&self, dev: Q::B) -> Result<()> {
        let args = self.0;
        let cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(&args.config)?)
            .with_context(|| format!("failed to read config {}", args.config.display()))?;
        let tokenizer_path = match args.tokenizer.clone() {
            Some(path) => path,
            None => {
                args.config.parent().context("config path has no parent")?.join("tokenizer.model")
            }
        };

        let t_load = Instant::now();
        let tokenizer = SpTokenizer::open(&tokenizer_path)?;
        let vb = model_helpers::load_weights::<Q>(&args.model, &dev)?;
        let model: TTSModel<Q> = TTSModel::load(&vb, Box::new(tokenizer), &cfg)?;
        vb.check_all_used_with_ignore(model_helpers::is_unused_by_tts_model)?;
        let voice_emb =
            model_helpers::load_voice_emb::<Q>(&args.voice, cfg.model_ext().as_deref(), &dev)?;
        let load_ms = ms(t_load.elapsed());

        // Tokenize up front: the loop needs the tokens anyway, and the KV cache is sized from
        // them. Long inputs are split into sentences, as `pocket_tts` does.
        let input = match args.lang.as_deref() {
            None => std::borrow::Cow::Borrowed(args.input.as_str()),
            Some(lang) => {
                use std::str::FromStr;
                let lang = ptts::preprocess::Lang::from_str(lang)?;
                std::borrow::Cow::Owned(ptts::preprocess::normalize_text(&args.input, lang))
            }
        };
        let chunks = ptts::tts_model::split_into_best_sentences(
            model.flow_lm.conditioner.tokenizer.as_deref().context("no tokenizer")?,
            &input,
            None,
        )?;
        let chunks = chunks
            .iter()
            .map(|chunk| {
                let (text, frames_after_eos) = ptts::tts_model::prepare_text_prompt(chunk);
                Ok((model.flow_lm.conditioner.tokenize(&text)?, frames_after_eos))
            })
            .collect::<Result<Vec<_>>>()?;

        // Condition on the voice once. Every iteration clones the resulting state, which is
        // what a server does per request, so the measurement is of generation rather than of
        // repeated voice conditioning.
        let voice_len = voice_emb.dim(1usize)?;
        let seq_budget = chunks
            .iter()
            .map(|(tokens, _)| voice_len + tokens.len() + max_frames_for(tokens.len()))
            .max()
            .unwrap_or(voice_len);
        let t_voice = Instant::now();
        let mut base_state = model.init_flow_lm_state(1, seq_budget)?;
        model.prompt_audio(&mut base_state, &voice_emb)?;
        let voice_ms = ms(t_voice.elapsed());

        let split = match args.split.as_deref() {
            Some(spec) => Some(parse_split(spec)?),
            None => None,
        };
        let generate = |m: &TTSModel<Q>, st: &TTSState<Q>| {
            if args.pipeline {
                one_pipelined(m, st, &chunks, args, split)
            } else {
                one(m, st, &chunks, args)
            }
        };
        for _ in 0..args.warmup {
            generate(&model, &base_state)?;
        }
        let mut runs = Vec::with_capacity(args.iters);
        for i in 0..args.iters {
            let r = generate(&model, &base_state)?;
            if args.per_iter {
                println!(
                    "iter {i:>3}: total {:>8.2}ms  ttfa {:>7.2}ms  frames {:>4}",
                    ms(r.total),
                    ms(r.ttfa),
                    r.frames.len()
                );
            }
            runs.push(r);
        }
        let first = &runs[0];
        let audio_ms = |r: &Run| r.samples as f64 / model.sample_rate() as f64 * 1e3;
        let totals: Vec<f64> = runs.iter().map(|r| ms(r.total)).collect();
        let ttfas: Vec<f64> = runs.iter().map(|r| ms(r.ttfa)).collect();
        // Pooled across iterations: per-frame variation matters more than which run it came
        // from, and one run has too few frames for a stable tail.
        let frames: Vec<f64> = runs.iter().flat_map(|r| r.frames.iter().copied().map(ms)).collect();
        // Audio produced per unit of wall time, so higher is faster than realtime.
        let rtfs: Vec<f64> = runs.iter().map(|r| audio_ms(r) / ms(r.total)).collect();

        println!();
        println!(
            "model {}  threads {}  input {} chars  audio {:.0}ms  frames/iter {}{}",
            args.model.display(),
            xn::get_num_threads(),
            input.len(),
            audio_ms(first),
            first.frames.len(),
            {
                let mut tags = Vec::new();
                if args.pipeline {
                    tags.push("pipelined".to_string());
                }
                if let Some((sample, decode)) = split {
                    tags.push(format!("split {sample}:{decode}"));
                }
                if args.mimi_batch != 1 {
                    tags.push(format!("mimi-batch {}", args.mimi_batch));
                }
                if tags.is_empty() { String::new() } else { format!("  [{}]", tags.join(", ")) }
            },
        );
        println!("load {load_ms:.1}ms, voice conditioning {voice_ms:.1}ms (both excluded below)");
        println!();
        println!(
            "{:<22} {:>5}  {:>9} {:>9} {:>9} {:>9} {:>9}",
            "metric", "n", "min", "mean", "p50", "p95", "max"
        );
        let sample_t: Vec<f64> =
            runs.iter().flat_map(|r| r.sample_t.iter().copied().map(ms)).collect();
        let decode_t: Vec<f64> =
            runs.iter().flat_map(|r| r.decode_t.iter().copied().map(ms)).collect();
        for (label, unit, prec, xs) in [
            ("total generate", "ms", 2, &totals),
            ("time to first audio", "ms", 2, &ttfas),
            ("per-frame", "ms", 3, &frames),
            ("  flow_lm sample", "ms", 3, &sample_t),
            ("  mimi decode", "ms", 3, &decode_t),
            ("rtf (higher is better)", "x realtime", 2, &rtfs),
        ] {
            row(label, unit, prec, &Stats::of(xs));
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    use std::str::FromStr;

    let args = Args::parse();
    if let Some(threads) = args.threads {
        // Must happen before the first tensor op, since it sets the size of rayon's global pool.
        xn::set_num_threads(threads);
    }
    let dtype = match args.quant.as_deref() {
        Some(quant) => xn::DTypeQ::from_str(quant)?,
        None if args.model.extension().and_then(|v| v.to_str()) == Some("gguf") => {
            anyhow::bail!("GGUF weights need an explicit --quant, e.g. --quant q8")
        }
        None => xn::DTypeQ::F32,
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        xn::with_avx(),
        xn::with_neon(),
        xn::with_simd128(),
        xn::with_f16c()
    );
    xn::Runner::new().cpu_only(args.cpu).dtype(dtype).run(Bench(&args), 0)?;
    Ok(())
}
