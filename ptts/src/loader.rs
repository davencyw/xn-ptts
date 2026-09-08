//! Reading checkpoints and voice embeddings off disk.
//!
//! Every frontend -- the examples, the ws-server, the Python bindings, the wasm build -- has to
//! rename the same checkpoint keys, skip the same unused tensors and unpack voice files the same
//! way. Keeping that here means a checkpoint layout change is one edit rather than four.

use crate::tts_model::TTSConfig;
use xn::nn::{Path, VB};
use xn::{Backend, BackendQ, Result, Tensor};

/// Maps upstream checkpoint names onto the names this crate's modules expect, dropping the
/// tensors the runtime has no use for.
pub fn remap_key(name: &str) -> Option<String> {
    // Skip keys we don't need.
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

/// Tensors that [`crate::tts_model::TTSModel::load`] legitimately leaves untouched: the encoder
/// side is only pulled in later by `MimiEnc::load`, and the quantizer is replaced by
/// [`crate::dummy_quantizer`].
///
/// Pass to `check_all_used_with_ignore` so a genuinely unused tensor is still an error.
pub fn is_unused_by_tts_model(name: &str) -> bool {
    name == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
        || name.starts_with("mimi.quantizer")
        || name.starts_with("mimi.encoder")
        || name.starts_with("speaker_mimi")
        || name == "flow_lm.speaker_proj_weight"
        // A prefix, not the single `conv.conv.weight` the examples used to name: the ws-server
        // already matched it this way, and taking the union keeps every caller as permissive as
        // it was.
        || name.starts_with("mimi.downsample.")
}

/// Loads GGUF or safetensors weights, picking the format from the extension.
pub fn load_weights<Q: BackendQ>(path: &std::path::Path, dev: &Q::B) -> Result<Path<Q::B>> {
    let vb = if path.extension().and_then(|v| v.to_str()) == Some("gguf") {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        VB::load_gguf_with_key_map(reader, dev.clone(), remap_key)?
    } else {
        VB::load_with_key_map(&[path], dev.clone(), remap_key)?
    };
    Ok(vb.root())
}

/// Loads a precomputed voice embedding as `[1, T, dim]`.
///
/// Voice files hold either `[T, dim]` or an already batched `[1, T, dim]`. When `model_ext` is
/// given and the file records one of its own, the two must agree -- a voice conditioned on a
/// different checkpoint produces confident nonsense rather than an error, so it is worth
/// catching here. Pass `None` to skip the check.
///
/// The result is f32 regardless of the backend's quantization; convert with `to::<Q::T>()`.
pub fn load_voice_emb<B: Backend>(
    path: &std::path::Path,
    model_ext: Option<&str>,
    dev: &B,
) -> Result<Tensor<f32, B>> {
    use xn::error::Context;

    let vb = VB::load(&[path], dev.clone())?;
    let names = vb.tensor_names();
    let key = names.first().context("no tensors found in voice embedding file")?;
    let shape = vb.shape(key).context("voice tensor not found")?;
    let dims = shape.dims().to_vec();
    let emb: Tensor<f32, B> = vb.tensor(key, shape)?;
    let emb = if dims.len() == 2 { emb.reshape((1, dims[0], dims[1]))? } else { emb };
    if let Some(model_ext) = model_ext {
        check_model_ext(path, model_ext)?;
    }
    Ok(emb)
}

/// Fails if the voice file records a `model_ext` other than `model_ext`. A file that records
/// none is accepted: older voices predate the metadata.
fn check_model_ext(path: &std::path::Path, model_ext: &str) -> Result<()> {
    let Some(voice_model_ext) = read_model_ext(path)? else { return Ok(()) };
    tracing::info!(?voice_model_ext, "voice embedding model_ext from metadata");
    if voice_model_ext != model_ext {
        xn::bail!(
            "voice embedding model_ext '{voice_model_ext}' does not match config model_ext '{model_ext}'"
        )
    }
    Ok(())
}

/// The `__metadata__.model_ext` string a voice file records, if any.
///
/// Reads the header JSON directly rather than going through
/// `SafeTensors::read_metadata`, which cannot be used here: it ends with
///
/// ```text
/// if buffer_end + 8 + n != buffer_len { return Err(MetadataIncompleteBuffer) }
/// ```
///
/// so it only accepts a buffer holding the tensor data as well, and rejects the
/// header alone for every file that has any. Only `__metadata__` is wanted, and
/// that lives in the header, so parse it there.
fn read_model_ext(path: &std::path::Path) -> Result<Option<String>> {
    #[derive(serde::Deserialize)]
    struct Header {
        #[serde(rename = "__metadata__", default)]
        metadata: std::collections::HashMap<String, String>,
    }

    let header = read_safetensors_header(path)?;
    let json = std::str::from_utf8(&header[8..])
        .map_err(|e| xn::Error::msg(format!("{}: header is not utf-8: {e}", path.display())))?;
    let header: Header = serde_json::from_str(json)
        .map_err(|e| xn::Error::msg(format!("{}: bad safetensors header: {e}", path.display())))?;
    Ok(header.metadata.get("model_ext").cloned())
}

/// Reads just enough of a safetensors file for `read_metadata`: the 8-byte
/// little-endian header length, then the header itself.
///
/// `load_voice_emb` is called once per voice while a model loads, and the
/// published checkpoint ships eight of them, so reading whole files here would
/// double the bytes touched at startup for no reason.
fn read_safetensors_header(path: &std::path::Path) -> Result<Vec<u8>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    // A file that is not safetensors can claim an absurd header length; refuse
    // to allocate on its word.
    if header_len > 100 * 1024 * 1024 {
        xn::bail!("{} does not look like a safetensors file", path.display())
    }
    let mut buf = len_bytes.to_vec();
    buf.resize(8 + header_len as usize, 0);
    file.read_exact(&mut buf[8..])?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a safetensors file holding one small tensor, plus whatever
    /// `__metadata__` is given, and returns its path.
    fn write_safetensors(
        dir: &std::path::Path,
        name: &str,
        metadata: Option<&str>,
    ) -> std::path::PathBuf {
        let data = [0u8; 16];
        let mut header = String::from("{");
        if let Some(metadata) = metadata {
            header.push_str(&format!("\"__metadata__\":{{\"model_ext\":\"{metadata}\"}},"));
        }
        header.push_str("\"emb\":{\"dtype\":\"F32\",\"shape\":[1,4],\"data_offsets\":[0,16]}}");
        let path = dir.join(name);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn voices_are_found_beside_the_weights_directory() {
        let root = std::env::temp_dir().join("ptts-voice-layout-test");
        std::fs::remove_dir_all(&root).ok();
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(root.join("voices")).unwrap();

        // A layout with the weights in `model/` and the voices its sibling.
        std::fs::write(model.join("model.q8.gguf"), b"not really a gguf").unwrap();
        write_safetensors(&root.join("voices"), "Freya.safetensors", None);
        write_safetensors(&root.join("voices"), "Zoey.safetensors", None);

        let artifacts = resolve_dir(&model, 0.7).unwrap();
        let names: Vec<&str> = artifacts.voices.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["Freya", "Zoey"]);

        // A voice beside the weights shadows one of the same name a level up.
        std::fs::create_dir_all(model.join("voices")).unwrap();
        write_safetensors(&model.join("voices"), "Freya.safetensors", None);
        let artifacts = resolve_dir(&model, 0.7).unwrap();
        let freya = artifacts.voices.iter().find(|(n, _)| n == "Freya").unwrap();
        assert!(freya.1.starts_with(&model), "expected {:?} under {:?}", freya.1, model);

        std::fs::remove_dir_all(&root).ok();
    }

    /// `check_model_ext` used to hand a header-only buffer to
    /// `SafeTensors::read_metadata`, which requires the tensor data to be
    /// present too and so rejected every voice file that had any. It went
    /// unnoticed because the only published config leaves `model_id` null,
    /// making `model_ext` `None` and skipping the check entirely.
    #[test]
    fn model_ext_is_read_from_a_header_that_has_tensor_data() {
        let dir = std::env::temp_dir().join("ptts-model-ext-test");
        std::fs::create_dir_all(&dir).unwrap();

        let tagged = write_safetensors(&dir, "tagged.safetensors", Some("a5e53131@800"));
        assert_eq!(read_model_ext(&tagged).unwrap().as_deref(), Some("a5e53131@800"));
        check_model_ext(&tagged, "a5e53131@800").unwrap();
        assert!(check_model_ext(&tagged, "deadbeef@1").is_err());

        // A file predating the metadata records none, and is accepted.
        let untagged = write_safetensors(&dir, "untagged.safetensors", None);
        assert_eq!(read_model_ext(&untagged).unwrap(), None);
        check_model_ext(&untagged, "a5e53131@800").unwrap();

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn remap_key_renames_and_drops() {
        // Dropped outright.
        for dropped in [
            "flow_lm.flow.w_s_t",
            "mimi.quantizer.vq.something",
            "mimi.quantizer.logvar_proj.weight",
        ] {
            assert_eq!(remap_key(dropped), None, "expected {dropped} to be dropped");
        }

        let cases = [
            (
                "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
                "flow_lm.speaker_proj_weight",
            ),
            (
                "flow_lm.condition_provider.conditioners.transcript_in_segment.emb.weight",
                "flow_lm.conditioner.emb.weight",
            ),
            (
                "flow_lm.backbone.layers.0.linear1.weight",
                "flow_lm.transformer.layers.0.linear1.weight",
            ),
            ("mimi.model.decoder.layers.0.weight", "mimi.decoder.layers.0.weight"),
        ];
        for (from, to) in cases {
            assert_eq!(remap_key(from).as_deref(), Some(to), "input: {from}");
        }

        // Anything unrecognized passes through untouched.
        assert_eq!(
            remap_key("mimi.decoder.layers.0.weight").as_deref(),
            Some("mimi.decoder.layers.0.weight")
        );
    }

    #[test]
    fn flow_lm_flow_is_renamed_not_dropped() {
        // `flow.w_s_t` is dropped but `flow.` alone is a rename; the two must not be confused.
        assert_eq!(
            remap_key("flow_lm.flow.layers.0.weight").as_deref(),
            Some("flow_lm.flow_net.layers.0.weight")
        );
    }

    #[test]
    fn unused_tensors_are_recognized() {
        for unused in [
            "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding",
            "mimi.quantizer.output_proj.weight",
            "mimi.encoder.layers.0.weight",
            "speaker_mimi.encoder.weight",
            "flow_lm.speaker_proj_weight",
            "mimi.downsample.conv.conv.weight",
            "mimi.downsample.something_else",
        ] {
            assert!(is_unused_by_tts_model(unused), "expected {unused} to be ignorable");
        }
        assert!(!is_unused_by_tts_model("flow_lm.transformer.layers.0.linear1.weight"));
        assert!(!is_unused_by_tts_model("mimi.decoder.layers.0.weight"));
    }
}

// ---------------------------------------------------------------------------
// Where a checkpoint comes from
// ---------------------------------------------------------------------------

/// Voice embeddings shipped with the published checkpoint.
pub const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

/// Default Hugging Face repo for the published weights.
pub const DEFAULT_REPO_ID: &str = "kyutai/pocket-tts";

/// Weight file names tried, in order, when the source does not name one.
pub const WEIGHT_CANDIDATES: &[&str] =
    &["model.safetensors", "model.q8.gguf", "tts_b6369a24.safetensors"];

/// Tokenizer file names tried, in order. `tokenizer.json` is the Hugging Face
/// `tokenizers` format, `tokenizer.model` is SentencePiece; [`crate::tok::Tok`]
/// picks the reader by extension.
pub const TOKENIZER_CANDIDATES: &[&str] = &["tokenizer.json", "tokenizer.model"];

/// Where to load a checkpoint from.
///
/// [`load_weights`] and [`load_voice_emb`] read files that someone has already
/// located; this decides *which* files, across the three layouts in use — the
/// published Hub repo, a local model directory, and explicit paths.
#[derive(Clone, Debug)]
pub enum ModelSource {
    /// A Hugging Face model repo. Requires the `hub` feature.
    Hub { repo_id: String, weights: Option<String> },
    /// A local directory holding `config.json`, a weights file, a tokenizer and
    /// an optional `voices/` or `embeddings/` subdirectory.
    Dir(std::path::PathBuf),
    /// Explicit file paths. `config` defaults to [`TTSConfig::v202601`] when absent.
    Files {
        config: Option<std::path::PathBuf>,
        weights: std::path::PathBuf,
        tokenizer: Option<std::path::PathBuf>,
    },
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
                    xn::bail!("weights file not found: {}", weights.display())
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

/// A checkpoint's files, located and its config parsed, ready to load.
#[derive(Clone, Debug)]
pub struct Artifacts {
    pub config: TTSConfig,
    pub weights: std::path::PathBuf,
    /// `None` when the source carries no tokenizer file; the caller must then
    /// supply a tokenizer itself.
    pub tokenizer: Option<std::path::PathBuf>,
    /// Voice name to embedding file, sorted by name.
    pub voices: Vec<(String, std::path::PathBuf)>,
}

fn read_config(path: &std::path::Path, temperature: f32) -> Result<TTSConfig> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| xn::Error::msg(format!("cannot read config {}: {e}", path.display())))?;
    let mut cfg: TTSConfig = serde_json::from_str(&text)
        .map_err(|e| xn::Error::msg(format!("cannot parse config {}: {e}", path.display())))?;
    cfg.temp = temperature;
    Ok(cfg)
}

fn resolve_dir(dir: &std::path::Path, temperature: f32) -> Result<Artifacts> {
    if !dir.is_dir() {
        xn::bail!("not a directory: {}", dir.display())
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

    // Voices are looked for beside the weights and then one level up: a
    // checkpoint is commonly laid out with `model/` holding the weights, config
    // and tokenizer, and `voices/` a sibling of it rather than a child. Pointing
    // `--dir` at the weights should still find them.
    let mut voices = vec![];
    let mut roots = vec![dir.to_path_buf()];
    if let Some(parent) = dir.parent() {
        roots.push(parent.to_path_buf());
    }
    for root in &roots {
        for sub in ["voices", "embeddings"] {
            collect_voice_dir(&root.join(sub), &mut voices);
        }
    }
    let default_voice = dir.join("default-voice.safetensors");
    if default_voice.is_file() {
        voices.push(("default".to_string(), default_voice));
    }
    // Retain before sorting: insertion order encodes precedence, so a voice
    // beside the weights wins over one of the same name a level up.
    let mut seen = std::collections::HashSet::new();
    voices.retain(|(name, _)| seen.insert(name.clone()));
    voices.sort();

    Ok(Artifacts { config, weights, tokenizer, voices })
}

/// Adds every `*.safetensors` file in `dir` to `voices`, keyed by file stem. A
/// missing or unreadable directory is not an error: voices are optional.
fn collect_voice_dir(dir: &std::path::Path, voices: &mut Vec<(String, std::path::PathBuf)>) {
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

    let config = match repo.get_optional("config.json") {
        Some(path) => read_config(&path, temperature)?,
        None => TTSConfig::v202601(temperature),
    };

    let weights = match weights {
        Some(name) => repo.get(name)?,
        None => match WEIGHT_CANDIDATES.iter().find_map(|name| repo.get_optional(name)) {
            Some(path) => path,
            None => xn::bail!(
                "no weights file in `{repo_id}`; expected one of {}",
                WEIGHT_CANDIDATES.join(", ")
            ),
        },
    };

    let tokenizer = TOKENIZER_CANDIDATES.iter().find_map(|name| repo.get_optional(name));

    let mut voices = vec![];
    for voice in VOICES {
        if let Some(path) = repo.get_optional(&format!("embeddings/{voice}.safetensors")) {
            voices.push((voice.to_string(), path));
        }
    }
    if let Some(path) = repo.get_optional("default-voice.safetensors") {
        voices.push(("default".to_string(), path));
    }
    voices.sort();

    Ok(Artifacts { config, weights, tokenizer, voices })
}

/// A Hugging Face model repo, wrapped so a download failure names the repo, the
/// file and the URL — `hf_hub`'s own errors mention none of the three, which
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

    fn get(&self, filename: &str) -> Result<std::path::PathBuf> {
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

    /// Like [`Self::get`] but maps any failure to `None`, for files that may
    /// legitimately be absent from a given repo layout.
    fn get_optional(&self, filename: &str) -> Option<std::path::PathBuf> {
        self.repo.get(filename).ok()
    }
}
