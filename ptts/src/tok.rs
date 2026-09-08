//! Tokenizer implementations for the [`crate::Tokenizer`] trait.
//!
//! Which one is available depends on the enabled features: `sp` pulls in
//! SentencePiece (the format the published `tokenizer.model` uses) and `hf`
//! pulls in Hugging Face `tokenizers` (`tokenizer.json`). Frontends that
//! tokenize elsewhere — `ptts-wasm` tokenizes in JavaScript — need neither and
//! can pass their own [`crate::Tokenizer`] to
//! [`crate::synth::SynthBuilder::tokenizer`].

#![cfg(any(feature = "sp", feature = "hf"))]

use crate::Tokenizer;
use std::path::Path;

/// A tokenizer loaded from a checkpoint file.
pub enum Tok {
    #[cfg(feature = "sp")]
    Sp(Box<sentencepiece::SentencePieceProcessor>),
    #[cfg(feature = "hf")]
    Hf(Box<tokenizers::Tokenizer>),
}

impl Tok {
    /// Load a tokenizer, choosing the implementation by file extension:
    /// `.model` is SentencePiece, anything else is `tokenizers` JSON.
    pub fn from_file(path: &Path) -> xn::Result<Self> {
        if path.extension().and_then(|e| e.to_str()) == Some("model") {
            Self::sentencepiece(path)
        } else {
            Self::huggingface(path)
        }
    }

    #[cfg(feature = "sp")]
    fn sentencepiece(path: &Path) -> xn::Result<Self> {
        let sp = sentencepiece::SentencePieceProcessor::open(path).map_err(|e| {
            xn::Error::msg(format!("cannot load SentencePiece tokenizer {}: {e}", path.display()))
        })?;
        Ok(Self::Sp(Box::new(sp)))
    }

    #[cfg(not(feature = "sp"))]
    fn sentencepiece(path: &Path) -> xn::Result<Self> {
        xn::bail!(
            "{} is a SentencePiece tokenizer; enable the `sp` feature of `ptts` to load it",
            path.display()
        )
    }

    #[cfg(feature = "hf")]
    fn huggingface(path: &Path) -> xn::Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            xn::Error::msg(format!("cannot load tokenizer {}: {e}", path.display()))
        })?;
        Ok(Self::Hf(Box::new(tok)))
    }

    #[cfg(not(feature = "hf"))]
    fn huggingface(path: &Path) -> xn::Result<Self> {
        xn::bail!(
            "{} is a Hugging Face tokenizer; enable the `hf` feature of `ptts` to load it",
            path.display()
        )
    }
}

impl Tokenizer for Tok {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        match self {
            #[cfg(feature = "sp")]
            Self::Sp(sp) => {
                Ok(sp.encode(text).map_err(xn::Error::wrap)?.into_iter().map(|v| v.id).collect())
            }
            #[cfg(feature = "hf")]
            Self::Hf(tok) => {
                Ok(tok.encode(text, false).map_err(xn::Error::wrap)?.get_ids().to_vec())
            }
        }
    }

    fn decode(&self, tokens: &[u32]) -> xn::Result<String> {
        match self {
            #[cfg(feature = "sp")]
            Self::Sp(sp) => sp.decode_piece_ids(tokens).map_err(xn::Error::wrap),
            #[cfg(feature = "hf")]
            Self::Hf(tok) => tok.decode(tokens, true).map_err(xn::Error::wrap),
        }
    }
}
