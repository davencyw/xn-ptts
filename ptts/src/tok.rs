//! A [`crate::Tokenizer`] backed by Hugging Face [`tokenizers`].
//!
//! The published checkpoints ship a SentencePiece `tokenizer.model`, which this crate does not
//! read: `scripts/convert-tokenizer.py` turns one into an equivalent `tokenizer.json` (same ids,
//! same round-trip, checked against `sentencepiece` as it converts) and [`Tok::open`] loads that.
//!
//! Available with the `hf` feature.

/// A Hugging Face tokenizer.
pub struct Tok(Box<tokenizers::Tokenizer>);

impl From<tokenizers::Tokenizer> for Tok {
    fn from(tok: tokenizers::Tokenizer) -> Self {
        Tok(Box::new(tok))
    }
}

impl Tok {
    /// Opens a `tokenizer.json`. A SentencePiece `.model` path — or a missing `tokenizer.json`
    /// sitting next to one — is refused with a pointer at the conversion script: every checkpoint
    /// has its own vocabulary, so there is no tokenizer to fall back to.
    pub fn open(path: &std::path::Path) -> xn::Result<Self> {
        if path.extension().and_then(|v| v.to_str()) == Some("model") {
            return Err(needs_conversion(path));
        }
        if !path.is_file() {
            let sp = path.with_file_name("tokenizer.model");
            if sp.is_file() {
                return Err(needs_conversion(&sp));
            }
        }
        tracing::info!(?path, "loading Hugging Face tokenizer");
        let tok = tokenizers::Tokenizer::from_file(path).map_err(xn::Error::wrap)?;
        Ok(Tok::from(tok))
    }
}

fn needs_conversion(sp: &std::path::Path) -> xn::Error {
    let sp = sp.display();
    xn::Error::msg(format!(
        "{sp} is a SentencePiece model, which ptts does not read; convert it once with \
         `uv run scripts/convert-tokenizer.py {sp}` and pass the tokenizer.json it writes"
    ))
    .with_path(sp.to_string())
}

impl crate::Tokenizer for Tok {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        let encoded = self.0.encode(text, false).map_err(xn::Error::wrap)?;
        Ok(encoded.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> xn::Result<String> {
        self.0.decode(ids, true).map_err(xn::Error::wrap)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sentencepiece_models_point_at_the_converter() {
        let err = super::Tok::open(std::path::Path::new("weights/tokenizer.model"))
            .err()
            .expect("a .model path is not a tokenizer.json");
        assert!(err.to_string().contains("convert-tokenizer.py"), "{err}");
    }
}
