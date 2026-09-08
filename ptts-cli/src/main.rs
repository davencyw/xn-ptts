//! Text to speech on the command line.
//!
//! ```text
//! cargo install ptts-cli
//! pocket-tts "hello world" -o out.wav
//! ```
//!
//! The crate is `ptts-cli` but the binary is `pocket-tts`: crates.io
//! `pocket-tts` belongs to an unrelated project.
//!
//! This is a front end and nothing else: every decision about how speech gets
//! made lives in `ptts::synth::Synth`. What is here is argument parsing, the
//! three forms `--voice` can take, and a one-line summary at the end.

use anyhow::Result;
use clap::Parser;
use ptts::synth::{DeviceKind, Quant, SpeechOptions, Synth};
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(name = "pocket-tts", version, about = "Generate speech from text using Pocket TTS")]
struct Args {
    /// Text to synthesize.
    text: String,

    /// Output WAV file path.
    #[arg(short, long, default_value = "output.wav")]
    output: std::path::PathBuf,

    /// Voice: a bundled voice id, a path to a voice `.safetensors`, or a path to
    /// a ~10s audio file to clone. Defaults to the first bundled voice.
    #[arg(short, long)]
    voice: Option<String>,

    /// List the voices this checkpoint ships and exit.
    #[arg(long)]
    list_voices: bool,

    /// Sampling temperature.
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Sampling seed.
    #[arg(short, long, default_value_t = 4242424242424242)]
    seed: u64,

    /// Load from a local directory holding config.json, weights, tokenizer and
    /// voices/ instead of downloading from the Hugging Face Hub.
    #[arg(long)]
    dir: Option<std::path::PathBuf>,

    /// Device to run on: auto, cpu, cuda, vulkan or metal.
    #[arg(long, default_value = "auto")]
    device: String,

    /// Weight format for the flow-LM linears, e.g. q8_0 or q4k. CPU only.
    #[arg(long)]
    quant: Option<String>,

    /// Classifier-free guidance coefficient. 1.0 disables it.
    #[arg(long)]
    cfg_coef: Option<f32>,

    /// Normalize the text for this language before tokenizing: `en`, `fr`, `de`, `es` or `pt`.
    #[arg(long)]
    lang: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing();

    let mut builder = Synth::builder()
        .device(DeviceKind::parse(&args.device)?)
        .temperature(args.temperature)
        .seed(args.seed);
    if let Some(dir) = args.dir.as_ref() {
        builder = builder.dir(dir);
    }
    if let Some(quant) = args.quant.as_deref() {
        builder = builder.quant(Quant::parse(quant)?);
    }
    if let Some(cfg_coef) = args.cfg_coef {
        builder = builder.cfg_coef(cfg_coef);
    }
    // An embedding file can be registered before the model loads; an audio file
    // has to wait until the speaker codec's sample rate is known.
    let voice = Voice::parse(args.voice.as_deref());
    if let Voice::Embedding(path) = &voice {
        builder = builder.add_voice(Voice::REGISTERED, path.clone());
    }

    tracing::info!("loading model");
    let mut tts = builder.build()?;
    tracing::info!(device = %tts.device_name(), weights = %tts.quant().as_str(), "model loaded");

    if args.list_voices {
        for name in tts.voices() {
            println!("{name}");
        }
        return Ok(());
    }

    let mut opts = SpeechOptions::default();
    match &voice {
        Voice::Default => {}
        Voice::Bundled(name) => opts = opts.voice(name.clone()),
        Voice::Embedding(_) => opts = opts.voice(Voice::REGISTERED),
        Voice::Audio(path) => {
            let pcm = ptts::audio::load_mono_at(path, tts.voice_prompt_sample_rate())?;
            tts.add_voice_from_pcm(Voice::REGISTERED, &pcm)?;
            opts = opts.voice(Voice::REGISTERED);
        }
    }

    // Text normalization is a property of the text, not of the model, so it
    // runs before anything is handed to `Synth`.
    let text = match args.lang.as_deref() {
        None => std::borrow::Cow::Borrowed(args.text.as_str()),
        Some(lang) => {
            let lang = ptts::preprocess::Lang::from_str(lang)?;
            std::borrow::Cow::Owned(ptts::preprocess::normalize_text(&args.text, lang))
        }
    };

    let start = std::time::Instant::now();
    let stream = tts.stream_with(&text, &opts)?;
    let sample_rate = stream.sample_rate();

    let mut pcm = Vec::new();
    let mut first_chunk = None;
    for chunk in stream {
        pcm.extend_from_slice(&chunk?);
        first_chunk.get_or_insert_with(|| start.elapsed());
    }

    let elapsed = start.elapsed().as_secs_f64();
    let duration = pcm.len() as f64 / sample_rate as f64;
    ptts::wav::write_wav_file(&args.output, &pcm, sample_rate as u32)?;
    println!(
        "wrote {} — {duration:.2}s of audio in {elapsed:.2}s (RTF {:.2}x, first audio after {:.0}ms)",
        args.output.display(),
        duration / elapsed,
        first_chunk.unwrap_or_default().as_secs_f64() * 1000.0,
    );
    Ok(())
}

/// What `--voice` was pointing at.
enum Voice {
    /// Not given: use whichever voice the model defaults to.
    Default,
    /// A voice id shipped with the checkpoint, e.g. `alba`.
    Bundled(String),
    /// A precomputed voice embedding, e.g. from the `create_voice` example.
    Embedding(std::path::PathBuf),
    /// An audio file to clone, ~10s of speech.
    Audio(std::path::PathBuf),
}

impl Voice {
    /// Name the two file forms get registered under.
    const REGISTERED: &'static str = "custom";

    fn parse(arg: Option<&str>) -> Self {
        match arg {
            None => Self::Default,
            Some(arg) if arg.ends_with(".safetensors") => Self::Embedding(arg.into()),
            Some(arg) if std::path::Path::new(arg).is_file() => Self::Audio(arg.into()),
            Some(arg) => Self::Bundled(arg.to_string()),
        }
    }
}

/// Progress goes to stderr via `tracing`; the result line goes to stdout. Set
/// `RUST_LOG=warn` for quiet, `RUST_LOG=debug` for more.
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, prelude::*};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::Layer::new().with_target(false).with_writer(std::io::stderr))
        .with(filter)
        .init();
}
