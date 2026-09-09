//! A [`crate::Tokenizer`] over the two tokenizer families the shipped checkpoints use.
//!
//! Which one a checkpoint wants is decided by its file: `tokenizer.model` is SentencePiece,
//! `tokenizer.json` is a Hugging Face `tokenizers` file. [`Tok::open`] picks by extension so
//! callers do not each reimplement that.
//!
//! Available with either the `sp` or `hf` feature; the corresponding variant only exists when
//! its feature is on.

/// A tokenizer of either supported family.
pub enum Tok {
    #[cfg(feature = "sp")]
    Sp(std::sync::Arc<sentencepiece::SentencePieceProcessor>),
    #[cfg(feature = "hf")]
    Hf(Box<tokenizers::Tokenizer>),
}

#[cfg(feature = "sp")]
impl From<sentencepiece::SentencePieceProcessor> for Tok {
    fn from(sp: sentencepiece::SentencePieceProcessor) -> Self {
        Tok::Sp(std::sync::Arc::new(sp))
    }
}

#[cfg(feature = "hf")]
impl From<tokenizers::Tokenizer> for Tok {
    fn from(tok: tokenizers::Tokenizer) -> Self {
        Tok::Hf(Box::new(tok))
    }
}

impl Tok {
    /// Opens a tokenizer, choosing the family from the extension: `.model` is SentencePiece,
    /// anything else is treated as a Hugging Face `tokenizers` file.
    pub fn open(path: &std::path::Path) -> xn::Result<Self> {
        use xn::error::Context;

        let path = path.to_str().context("invalid tokenizer path")?;
        if path.ends_with(".model") { Self::open_sp(path) } else { Self::open_hf(path) }
    }

    #[cfg(feature = "sp")]
    fn open_sp(path: &str) -> xn::Result<Self> {
        tracing::info!(?path, "loading SentencePiece tokenizer");
        let sp = sentencepiece::SentencePieceProcessor::open(path).map_err(xn::Error::wrap)?;
        Ok(Tok::from(sp))
    }

    #[cfg(not(feature = "sp"))]
    fn open_sp(path: &str) -> xn::Result<Self> {
        xn::bail!("{path} needs a SentencePiece tokenizer; rebuild ptts with the 'sp' feature")
    }

    #[cfg(feature = "hf")]
    fn open_hf(path: &str) -> xn::Result<Self> {
        tracing::info!(?path, "loading Hugging Face tokenizer");
        let tok = tokenizers::Tokenizer::from_file(path).map_err(xn::Error::wrap)?;
        Ok(Tok::from(tok))
    }

    #[cfg(not(feature = "hf"))]
    fn open_hf(path: &str) -> xn::Result<Self> {
        xn::bail!("{path} needs a Hugging Face tokenizer; rebuild ptts with the 'hf' feature")
    }
}

impl crate::Tokenizer for Tok {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        let tokens = match self {
            #[cfg(feature = "sp")]
            Tok::Sp(sp) => {
                sp.encode(text).map_err(xn::Error::wrap)?.into_iter().map(|v| v.id).collect()
            }
            #[cfg(feature = "hf")]
            Tok::Hf(tok) => tok.encode(text, false).map_err(xn::Error::wrap)?.get_ids().to_vec(),
        };
        Ok(tokens)
    }

    fn decode(&self, ids: &[u32]) -> xn::Result<String> {
        let decoded = match self {
            #[cfg(feature = "sp")]
            Tok::Sp(sp) => sp.decode_piece_ids(ids).map_err(xn::Error::wrap)?,
            #[cfg(feature = "hf")]
            Tok::Hf(tok) => tok.decode(ids, true).map_err(xn::Error::wrap)?,
        };
        Ok(decoded)
    }
}
