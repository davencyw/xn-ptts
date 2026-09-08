//! Locating and loading a Pocket TTS checkpoint.
//!
//! The published weight names do not match this crate's module tree, so every
//! load goes through [`remap_key`]. That mapping is part of the checkpoint
//! format rather than a detail of any one frontend, and it used to be copied
//! verbatim into five places (the `pocket_tts` and `create_voice` examples,
//! `ptts-pyo3`, `ptts-wasm` and `ptts-ws-server`). It lives here now.

use crate::tts_model::TTSConfig;
use std::path::{Path as FsPath, PathBuf};
use xn::nn::VB;
use xn::{Backend, Result, Tensor};

/// Voice embeddings shipped with the published checkpoint.
pub const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

/// Default Hugging Face repo for the published weights.
pub const DEFAULT_REPO_ID: &str = "kyutai/pocket-tts";

/// Weight file names tried, in order, when the source does not name one.
pub const WEIGHT_CANDIDATES: &[&str] =
    &["model.safetensors", "model.q8.gguf", "tts_b6369a24.safetensors"];

/// Tokenizer file names tried, in order. `tokenizer.json` is the Hugging Face
/// `tokenizers` format; `tokenizer.model` is SentencePiece.
pub const TOKENIZER_CANDIDATES: &[&str] = &["tokenizer.json", "tokenizer.model"];

/// Rewrite a published tensor name to the name this crate loads it under, or
/// `None` to drop the tensor entirely.
///
/// Dropped tensors belong to parts of the reference model this runtime does not
/// use: the flow's `w_s_t` buffer and the real Mimi quantizer (replaced by
/// [`crate::dummy_quantizer`]).
pub fn remap_key(name: &str) -> Option<String> {
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
    {
        return None;
    }

    let mut name = name.to_string();

    // Order matters: more specific replacements first.
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");

    Some(name)
}

/// True for tensors a checkpoint may legitimately contain without this runtime
/// consuming them, so that [`VB::check_all_used_with_ignore`] does not reject a
/// valid file.
///
/// The Mimi encoder is only loaded for voice cloning, the Mimi quantizer is
/// replaced by [`crate::dummy_quantizer`], and the speaker projection is loaded
/// by [`crate::tts_model::MimiEnc`] rather than the decoder path. This is the
/// union of the ignore lists the frontends had drifted into keeping separately.
pub fn is_optional_tensor(name: &str) -> bool {
    name == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
        || name == "flow_lm.speaker_proj_weight"
        || name.starts_with("mimi.encoder")
        || name.starts_with("mimi.quantizer")
        || name.starts_with("mimi.downsample.")
        // A checkpoint with a dedicated speaker codec (`TTSConfig::speaker_mimi`) ships it
        // under its own prefix; only its encoder is ever loaded, and only for voice cloning.
        || name.starts_with("speaker_mimi")
}

/// Open `path` as a var builder, dispatching on the file extension: `.gguf`
/// files are read through the GGUF reader, anything else as safetensors.
pub fn load_var_builder<B: Backend>(path: &FsPath, device: B) -> Result<VB<B>> {
    if path.extension().and_then(|v| v.to_str()) == Some("gguf") {
        let reader = std::fs::File::open(path).map_err(|e| {
            xn::Error::msg(format!("cannot open weights at {}: {e}", path.display()))
        })?;
        VB::load_gguf_with_key_map(std::io::BufReader::new(reader), device, remap_key)
    } else {
        VB::load_with_key_map(&[path], device, remap_key)
    }
}

/// Load a voice-conditioning embedding from a safetensors file, normalizing it
/// to `[1, frames, dim]`.
///
/// The file is expected to hold a single tensor; its name is not significant.
pub fn load_voice_embedding<B: Backend>(path: &FsPath, device: &B) -> Result<Tensor<f32, B>> {
    let vb = VB::load(&[path], device.clone())?;
    let names = vb.tensor_names();
    let key = match names.first() {
        Some(key) => *key,
        None => xn::bail!("no tensors in voice embedding file {}", path.display()),
    };
    let shape = match vb.shape(key) {
        Some(shape) => shape.clone(),
        None => xn::bail!("voice tensor {key} not found in {}", path.display()),
    };
    let dims = shape.dims().to_vec();
    let emb: Tensor<f32, B> = vb.tensor(key, shape)?;
    match dims.as_slice() {
        [frames, dim] => emb.reshape((1, *frames, *dim)),
        [_, _, _] => Ok(emb),
        other => {
            xn::bail!("expected a voice embedding of rank 2 or 3, got {other:?} in {key}")
        }
    }
}

/// The `model_ext` recorded in a voice file's safetensors metadata, if any.
///
/// A voice embedding is only valid for the checkpoint it was computed from;
/// [`crate::synth::SynthBuilder`] compares this against
/// [`TTSConfig::model_ext`] and refuses a mismatch.
pub fn voice_model_ext(path: &FsPath) -> Result<Option<String>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| {
        xn::Error::msg(format!("cannot read voice embedding {}: {e}", path.display()))
    })?;
    // A safetensors file starts with the header length as a little-endian u64,
    // so read just the header rather than the whole tensor payload — this runs
    // once per voice at load time.
    let mut len_bytes = [0u8; 8];
    if file.read_exact(&mut len_bytes).is_err() {
        return Ok(None);
    }
    let header_len = u64::from_le_bytes(len_bytes);
    // Guard against a corrupt or non-safetensors file claiming a huge header.
    if header_len > 100 * 1024 * 1024 {
        xn::bail!("{} does not look like a safetensors file", path.display());
    }
    let mut buf = len_bytes.to_vec();
    buf.resize(8 + header_len as usize, 0);
    if file.read_exact(&mut buf[8..]).is_err() {
        return Ok(None);
    }
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&buf).map_err(xn::Error::wrap)?;
    Ok(metadata.metadata().as_ref().and_then(|m| m.get("model_ext").cloned()))
}

/// Where to load a checkpoint from.
#[derive(Clone, Debug)]
pub enum ModelSource {
    /// A Hugging Face model repo. Requires the `hub` feature.
    Hub { repo_id: String, weights: Option<String> },
    /// A local directory holding `config.json`, a weights file, a tokenizer and
    /// an optional `voices/` or `embeddings/` subdirectory.
    Dir(PathBuf),
    /// Explicit file paths. `config` defaults to [`TTSConfig::v202601`] when absent.
    Files { config: Option<PathBuf>, weights: PathBuf, tokenizer: Option<PathBuf> },
}

impl ModelSource {
    /// The published checkpoint on the Hugging Face Hub.
    pub fn hub() -> Self {
        Self::Hub { repo_id: DEFAULT_REPO_ID.to_string(), weights: None }
    }

    /// A specific Hugging Face repo, with the default weight-file search order.
    pub fn hub_repo(repo_id: impl Into<String>) -> Self {
        Self::Hub { repo_id: repo_id.into(), weights: None }
    }
}

/// A checkpoint's files, located and its config parsed, ready to load.
#[derive(Clone, Debug)]
pub struct Artifacts {
    pub config: TTSConfig,
    pub weights: PathBuf,
    /// `None` when the source carries no tokenizer file; the caller must then
    /// supply a tokenizer itself.
    pub tokenizer: Option<PathBuf>,
    /// Voice name to embedding file, sorted by name.
    pub voices: Vec<(String, PathBuf)>,
}

impl ModelSource {
    /// Locate every file this source provides, downloading if needed, and parse
    /// the config.
    pub fn resolve(&self, temperature: f32) -> Result<Artifacts> {
        match self {
            Self::Hub { repo_id, weights } => resolve_hub(repo_id, weights.as_deref(), temperature),
            Self::Dir(dir) => resolve_dir(dir, temperature),
            Self::Files { config, weights, tokenizer } => {
                let config = match config {
                    Some(path) => read_config(path, temperature)?,
                    None => TTSConfig::v202601(temperature),
                };
                if !weights.is_file() {
                    xn::bail!("weights file not found: {}", weights.display());
                }
                Ok(Artifacts {
                    config,
                    weights: weights.clone(),
                    tokenizer: tokenizer.clone(),
                    voices: vec![],
                })
            }
        }
    }
}

fn read_config(path: &FsPath, temperature: f32) -> Result<TTSConfig> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| xn::Error::msg(format!("cannot read config {}: {e}", path.display())))?;
    let mut cfg: TTSConfig = serde_json::from_str(&text)
        .map_err(|e| xn::Error::msg(format!("cannot parse config {}: {e}", path.display())))?;
    cfg.temp = temperature;
    Ok(cfg)
}

fn resolve_dir(dir: &FsPath, temperature: f32) -> Result<Artifacts> {
    if !dir.is_dir() {
        xn::bail!("not a directory: {}", dir.display());
    }
    let config_path = dir.join("config.json");
    let config = if config_path.is_file() {
        read_config(&config_path, temperature)?
    } else {
        TTSConfig::v202601(temperature)
    };

    let weights = WEIGHT_CANDIDATES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| {
        xn::Error::msg(format!(
            "no weights file in {}; expected one of {}",
            dir.display(),
            WEIGHT_CANDIDATES.join(", ")
        ))
    })?;

    let tokenizer =
        TOKENIZER_CANDIDATES.iter().map(|name| dir.join(name)).find(|path| path.is_file());

    let mut voices = vec![];
    for sub in ["voices", "embeddings"] {
        collect_voice_dir(&dir.join(sub), &mut voices);
    }
    let default_voice = dir.join("default-voice.safetensors");
    if default_voice.is_file() {
        voices.push(("default".to_string(), default_voice));
    }
    voices.sort();
    voices.dedup_by(|a, b| a.0 == b.0);

    Ok(Artifacts { config, weights, tokenizer, voices })
}

/// Add every `*.safetensors` file in `dir` to `voices`, keyed by file stem.
/// A missing or unreadable directory is not an error — voices are optional.
fn collect_voice_dir(dir: &FsPath, voices: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
            voices.push((name.to_string(), path));
        }
    }
}

#[cfg(not(feature = "hub"))]
fn resolve_hub(_: &str, _: Option<&str>, _: f32) -> Result<Artifacts> {
    xn::bail!(
        "loading from the Hugging Face Hub requires the `hub` feature of the `ptts` crate; \
         use ModelSource::Dir or ModelSource::Files to load local files instead"
    )
}

#[cfg(feature = "hub")]
fn resolve_hub(repo_id: &str, weights: Option<&str>, temperature: f32) -> Result<Artifacts> {
    let repo = HubRepo::open(repo_id)?;

    let config = match repo.get_optional("config.json")? {
        Some(path) => read_config(&path, temperature)?,
        None => TTSConfig::v202601(temperature),
    };

    let weights = match weights {
        Some(name) => repo.get(name)?,
        None => {
            let mut found = None;
            for name in WEIGHT_CANDIDATES {
                if let Some(path) = repo.get_optional(name)? {
                    found = Some(path);
                    break;
                }
            }
            match found {
                Some(path) => path,
                None => xn::bail!(
                    "no weights file in `{repo_id}`; expected one of {}",
                    WEIGHT_CANDIDATES.join(", ")
                ),
            }
        }
    };

    let mut tokenizer = None;
    for name in TOKENIZER_CANDIDATES {
        if let Some(path) = repo.get_optional(name)? {
            tokenizer = Some(path);
            break;
        }
    }

    let mut voices = vec![];
    for voice in VOICES {
        if let Some(path) = repo.get_optional(&format!("embeddings/{voice}.safetensors"))? {
            voices.push((voice.to_string(), path));
        }
    }
    if let Some(path) = repo.get_optional("default-voice.safetensors")? {
        voices.push(("default".to_string(), path));
    }
    voices.sort();

    Ok(Artifacts { config, weights, tokenizer, voices })
}

/// A Hugging Face model repo, wrapped so download failures name the repo, the
/// file and the URL. `hf_hub`'s own errors mention none of the three, which
/// makes a gated repo or a renamed file hard to diagnose.
#[cfg(feature = "hub")]
struct HubRepo {
    repo: hf_hub::api::sync::ApiRepo,
    repo_id: String,
}

#[cfg(feature = "hub")]
impl HubRepo {
    fn open(repo_id: &str) -> Result<Self> {
        use hf_hub::{Repo, RepoType, api::sync::Api};
        let api = Api::new()
            .map_err(|e| xn::Error::msg(format!("cannot reach the Hugging Face Hub: {e}")))?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
        Ok(Self { repo, repo_id: repo_id.to_string() })
    }

    fn get(&self, filename: &str) -> Result<PathBuf> {
        self.repo.get(filename).map_err(|e| {
            let url = self.repo.url(filename);
            xn::Error::msg(format!(
                "failed to fetch `{filename}` from `{}` ({url}): {e}\n\
                 If the repo is gated, accept its terms on huggingface.co and run \
                 `huggingface-cli login` (or set HF_TOKEN).",
                self.repo_id
            ))
        })
    }

    /// Like [`Self::get`] but maps a failure to `None`. Used for files that may
    /// legitimately be absent from a given repo layout.
    fn get_optional(&self, filename: &str) -> Result<Option<PathBuf>> {
        Ok(self.repo.get(filename).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_key_drops_unused_tensors() {
        assert_eq!(remap_key("flow_lm.flow.w_s_t"), None);
        assert_eq!(remap_key("mimi.quantizer.vq.codebook"), None);
        assert_eq!(remap_key("mimi.quantizer.logvar_proj.weight"), None);
    }

    #[test]
    fn remap_key_rewrites_published_names() {
        assert_eq!(
            remap_key("flow_lm.backbone.layers.0.linear1.weight").as_deref(),
            Some("flow_lm.transformer.layers.0.linear1.weight")
        );
        assert_eq!(
            remap_key("flow_lm.flow.layers.0.weight").as_deref(),
            Some("flow_lm.flow_net.layers.0.weight")
        );
        assert_eq!(
            remap_key("mimi.model.decoder.0.weight").as_deref(),
            Some("mimi.decoder.0.weight")
        );
        assert_eq!(
            remap_key("flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight")
                .as_deref(),
            Some("flow_lm.conditioner.embed.weight")
        );
        assert_eq!(
            remap_key("flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight")
                .as_deref(),
            Some("flow_lm.speaker_proj_weight")
        );
    }

    #[test]
    fn remap_key_leaves_other_names_alone() {
        assert_eq!(remap_key("mimi.decoder.1.bias").as_deref(), Some("mimi.decoder.1.bias"));
    }

    #[test]
    fn optional_tensors_cover_every_frontend_ignore_list() {
        // The union of what the example and ptts-ws-server each ignored.
        for name in [
            "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding",
            "flow_lm.speaker_proj_weight",
            "mimi.encoder.0.weight",
            "mimi.quantizer.output_proj.weight",
            "mimi.downsample.conv.conv.weight",
            // Added by the separate-speaker-Mimi checkpoints; upstream's
            // `model_helpers::is_unused_by_tts_model` carries this arm too.
            "speaker_mimi.decoder.0.weight",
            "speaker_mimi.quantizer.output_proj.weight",
        ] {
            assert!(is_optional_tensor(name), "{name} should be optional");
        }
        assert!(!is_optional_tensor("flow_lm.transformer.layers.0.linear1.weight"));
        assert!(!is_optional_tensor("mimi.decoder.0.weight"));
    }
}
