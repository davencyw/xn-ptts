//! Precomputes voice conditioning into KV-cache voice files.
//!
//! An embedding-style voice has to be run through `prompt_audio` before it can
//! be used, which is a 125-frame prefill through the whole backbone. Doing that
//! on a phone, once per voice, at page load, spends the device's thermal budget
//! immediately before the user asks for any speech. The conditioning is a pure
//! function of the model and the embedding, so it belongs here instead.
//!
//! The output matches the format `ptts-wasm` already loads directly:
//! `transformer.layers.{i}.self_attn/cache` shaped `[2, B, T, H, D]` (k stacked
//! on v) plus `current_end`.

use anyhow::{Context, Result};
use ptts::tts_model::{TTSConfig, TTSModel};
use ptts::transformer::LayerAttentionState;
use std::collections::HashMap;
use xn::quantized::Q80F32;
use xn::{CPU, Tensor, TypedTensor};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let repo = std::path::PathBuf::from(
        args.next().unwrap_or("/Users/david/Documents/code/phonon-inference".into()),
    );
    let out = std::path::PathBuf::from(
        args.next().unwrap_or_else(|| repo.join("voices-prepared").to_string_lossy().into()),
    );
    let model_dir = repo.join("model");

    let cfg: TTSConfig =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    let model_ext = cfg.model_id.as_ref().map(|m| format!("{}@{}", m.sig, m.epoch));
    let root = ptts::loader::load_weights::<Q80F32>(&model_dir.join("model.q8.gguf"), &CPU)?;
    let tok: Box<dyn ptts::Tokenizer + Send + Sync> =
        Box::new(ptts::tok::Tok::open(&model_dir.join("tokenizer.model"))?);
    let model = TTSModel::<Q80F32>::load(&root, tok, &cfg)?;
    println!("model loaded ({} layers)", cfg.flow_lm.num_layers);

    std::fs::create_dir_all(&out)?;
    let mut voices: Vec<_> = std::fs::read_dir(repo.join("voices"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    voices.sort();

    for path in &voices {
        let name = path.file_stem().context("no stem")?.to_string_lossy().to_string();
        let emb = ptts::loader::load_voice_emb(path, model_ext.as_deref(), &CPU)?;
        let (_, frames, _) = emb.dims3()?;

        // `prompt_audio` appends exactly `frames` entries, so that is the budget.
        let mut state = model.init_flow_lm_state(1, frames)?;
        model.prompt_audio(&mut state, &emb)?;

        let mut tensors: HashMap<String, TypedTensor<xn::CpuDevice>> = HashMap::new();
        let mut layers = 0usize;
        for (i, layer) in state.flow_lm_state.transformer_state.layer_states.iter().enumerate() {
            let LayerAttentionState::FlowLm(mha) = layer else {
                anyhow::bail!("layer {i} is not a flow-LM attention layer");
            };
            let end = mha.current_end;
            // Trim the budget down to what was actually written, then stack k on
            // v so one tensor per layer carries both, as the loader expects.
            let k = mha.k_cache.narrow(1, 0..end)?.contiguous()?;
            let v = mha.v_cache.narrow(1, 0..end)?.contiguous()?;
            let (b, t, h, d) = k.dims4()?;
            let k = k.reshape((1, b, t, h, d))?;
            let v = v.reshape((1, b, t, h, d))?;
            tensors.insert(
                format!("transformer.layers.{i}.self_attn/cache"),
                TypedTensor::F32(Tensor::cat(&[&k, &v], 0)?),
            );
            tensors.insert(
                format!("transformer.layers.{i}.self_attn/current_end"),
                TypedTensor::F32(Tensor::from_vec(vec![end as f32; end], (end,), &CPU)?),
            );
            layers += 1;
        }

        let dst = out.join(format!("{name}.safetensors"));
        xn::safetensors::save(&tensors, &dst)?;
        let mb = std::fs::metadata(&dst)?.len() as f64 / 1e6;
        println!("  {name}: {layers} layers, {frames} frames -> {mb:.1} MB");
    }
    println!("\nwrote {} voices to {}", voices.len(), out.display());
    Ok(())
}
