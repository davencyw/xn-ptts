//! Bits the examples share that are not worth a place in the library.
//!
//! Weight loading, key remapping and voice-embedding loading now live in `ptts::loader`, and
//! the tokenizer in `ptts::tok`; this re-exports them so each example has one place to look.
#![allow(dead_code, unused_imports)]

pub use ptts::loader::{is_unused_by_tts_model, load_voice_emb, load_weights, remap_key};
#[cfg(feature = "sp")]
pub use ptts::tok::Tok;

/// Frames an utterance of `num_tokens` tokens is allowed to generate before it is cut off.
pub fn max_frames_for(num_tokens: usize) -> usize {
    ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize
}
