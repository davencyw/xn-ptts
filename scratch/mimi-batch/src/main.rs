//! Does decoding several Mimi frames per call beat one at a time?
//!
//! With T=1 the col2im path inside `conv_transpose1d` runs a gemm with m=1 --
//! a matrix-vector product, so it is bandwidth-bound and reads the whole
//! kernel per frame. Batching should turn each of those into a real matmul.

use anyhow::Result;
use ptts::tts_model::{TTSConfig, TTSModel};
use std::time::Instant;
use xn::quantized::Q80F32;
use xn::{CPU, Tensor};

fn main() -> Result<()> {
    // wasm runs single-threaded, which is the condition that matters here.
    if std::env::var("THREADS").as_deref() != Ok("many") {
        xn::set_num_threads(1);
    }
    let dir = std::path::PathBuf::from("/Users/david/Documents/code/phonon-inference/model");
    let cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let ldim = cfg.flow_lm.ldim;

    // Goes through the library loader so GGUF key remapping matches production.
    let root = ptts::loader::load_weights::<Q80F32>(&dir.join("model.q8.gguf"), &CPU)?;
    let tok: Box<dyn ptts::Tokenizer + Send + Sync> =
        Box::new(ptts::tok::Tok::open(&dir.join("tokenizer.model"))?);
    let model = TTSModel::<Q80F32>::load(&root, tok, &cfg)?;

    // Latents standing in for sampled ones: this measures the decoder, and its
    // cost does not depend on the values.
    let n = 64usize;
    let latents: Vec<Tensor<f32, _>> = (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..ldim).map(|j| ((i * ldim + j) as f32 * 0.01).sin() * 0.5).collect();
            Tensor::from_vec(v, (1, 1, ldim), &CPU).unwrap()
        })
        .collect();

    println!("threads = {}", xn::get_num_threads());
    println!("{:>6}  {:>10}  {:>12}  {:>9}  {:>12}", "chunk", "total ms", "ms/frame", "speedup", "vs chunk=1");
    let mut baseline = 0.0f64;
    let mut reference: Vec<f32> = vec![];
    for chunk in [1usize, 2, 4, 8, 16] {
        // Fresh state per run: streaming state carries across calls.
        let mut st = model.init_mimi_state(1)?;
        // Warm up so the first run does not pay for lazily-built scratch.
        let warm = Tensor::cat(&latents[..chunk].iter().collect::<Vec<_>>(), 1)?;
        let _ = model.decode_latent(&warm, &mut st)?;

        let mut st = model.init_mimi_state(1)?;
        let t = Instant::now();
        let mut frames = 0usize;
        let mut pcm: Vec<f32> = vec![];
        for group in latents.chunks(chunk) {
            let refs: Vec<&Tensor<f32, _>> = group.iter().collect();
            let batched = Tensor::cat(&refs, 1)?;
            let out = model.decode_latent(&batched, &mut st)?;
            frames += group.len();
            pcm.extend(out.narrow(0, 0..1)?.contiguous()?.to_vec()?);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let per = ms / frames as f64;
        if chunk == 1 {
            baseline = per;
            reference = pcm.clone();
        }
        // Batching must not change the audio: the streaming convolution state
        // is supposed to make a chunk of k frames identical to k single frames.
        let diff = if pcm.len() != reference.len() {
            format!("LEN {} vs {}", pcm.len(), reference.len())
        } else {
            let max = pcm
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            format!("max|d| {max:.2e}")
        };
        println!("{chunk:>6}  {ms:>10.1}  {per:>12.3}  {:>8.2}x  {diff:>12}", baseline / per);
    }
    Ok(())
}
