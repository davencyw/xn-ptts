//! Python bindings for Pocket TTS.
//!
//! ```python
//! import ptts
//!
//! tts = ptts.TTS()
//! tts.save("out.wav", "Hello world")
//! ```
//!
//! The whole pipeline is `ptts::synth::Synth`, so this file is a translation
//! layer and nothing more: numpy in and out, GIL released around the slow
//! parts, and Ctrl-C handled between audio chunks.

use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use ptts::synth::{DeviceKind, Quant, SpeechOptions, SpeechStream, Synth};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};

/// Map a `ptts` error onto a Python exception. `ValueError` throughout: every
/// failure here is a bad argument, a missing file or a bad checkpoint.
trait IntoPy<R> {
    fn py(self) -> PyResult<R>;
}

impl<R, E: Into<xn::Error>> IntoPy<R> for Result<R, E> {
    fn py(self) -> PyResult<R> {
        self.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.into().to_string()))
    }
}

/// Flatten a conditioning embedding of shape `[T, dim]` or `[1, T, dim]` into
/// contiguous values plus its frame count and width.
fn embedding_dims(arr: &PyReadonlyArrayDyn<'_, f32>) -> PyResult<(Vec<f32>, usize, usize)> {
    let (frames, dim) = match arr.shape() {
        [t, d] => (*t, *d),
        [1, t, d] => (*t, *d),
        shape => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected an embedding of shape [T, dim] or [1, T, dim], got {shape:?}"
            )));
        }
    };
    let values = arr.as_array().iter().copied().collect();
    Ok((values, frames, dim))
}

/// A loaded Pocket TTS model.
///
/// Constructing one downloads the checkpoint on first use and caches it the way
/// the `huggingface_hub` CLI does.
#[pyclass(name = "TTS", module = "ptts")]
struct Tts {
    inner: Arc<Mutex<Synth>>,
}

#[pymethods]
impl Tts {
    /// `TTS(dir=None, voice=None, device=None, quant=None, temperature=0.7, seed=..., cfg_coef=None, eos_threshold=None)`
    #[new]
    #[pyo3(signature = (
        dir = None,
        voice = None,
        device = None,
        quant = None,
        temperature = 0.7,
        seed = 4242424242424242,
        cfg_coef = None,
        eos_threshold = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        dir: Option<std::path::PathBuf>,
        voice: Option<String>,
        device: Option<&str>,
        quant: Option<&str>,
        temperature: f32,
        seed: u64,
        cfg_coef: Option<f32>,
        eos_threshold: Option<f32>,
    ) -> PyResult<Self> {
        let device = match device {
            None => DeviceKind::Auto,
            Some(name) => DeviceKind::parse(name).py()?,
        };
        let quant = match quant {
            None => Quant::F32,
            Some(name) => Quant::parse(name).py()?,
        };
        // Loading reads hundreds of megabytes and runs no Python; hold no GIL.
        py.detach(move || {
            let mut builder =
                Synth::builder().device(device).quant(quant).temperature(temperature).seed(seed);
            if let Some(dir) = dir {
                builder = builder.dir(dir);
            }
            if let Some(voice) = voice {
                builder = builder.voice(voice);
            }
            if let Some(cfg_coef) = cfg_coef {
                builder = builder.cfg_coef(cfg_coef);
            }
            if let Some(eos_threshold) = eos_threshold {
                builder = builder.eos_threshold(eos_threshold);
            }
            let synth = builder.build().py()?;
            Ok(Self { inner: Arc::new(Mutex::new(synth)) })
        })
    }

    /// Sample rate of the audio this model produces, in Hz.
    #[getter]
    fn sample_rate(&self) -> PyResult<usize> {
        Ok(self.lock()?.sample_rate())
    }

    /// Names of the registered voices, sorted.
    #[getter]
    fn voices(&self) -> PyResult<Vec<String>> {
        Ok(self.lock()?.voices())
    }

    /// Device the model is running on, e.g. `"cpu"`.
    #[getter]
    fn device(&self) -> PyResult<String> {
        Ok(self.lock()?.device_name())
    }

    /// Weight format actually loaded, e.g. `"q8_0"`.
    #[getter]
    fn quant(&self) -> PyResult<&'static str> {
        Ok(self.lock()?.quant().as_str())
    }

    /// Sample rate that `clone_voice` expects its PCM in, in Hz.
    #[getter]
    fn voice_prompt_sample_rate(&self) -> PyResult<usize> {
        Ok(self.lock()?.voice_prompt_sample_rate())
    }

    /// Synthesize `text` and return the waveform as a 1-D float32 array.
    #[pyo3(signature = (text, *, voice=None, temperature=None, seed=None, cfg_coef=None))]
    fn synth<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let opts = options(voice, temperature, seed, cfg_coef);
        let stream = self.lock()?.stream_with(text, &opts).py()?;
        let pcm = drain(py, stream)?;
        Ok(PyArray1::from_vec(py, pcm))
    }

    /// Synthesize `text` straight to a mono 16-bit WAV file at `path`.
    #[pyo3(signature = (path, text, *, voice=None, temperature=None, seed=None, cfg_coef=None))]
    #[allow(clippy::too_many_arguments)]
    fn save(
        &self,
        py: Python<'_>,
        path: std::path::PathBuf,
        text: &str,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<f64> {
        let opts = options(voice, temperature, seed, cfg_coef);
        let (pcm, sample_rate) = {
            let synth = self.lock()?;
            let stream = synth.stream_with(text, &opts).py()?;
            let sample_rate = stream.sample_rate();
            (drain(py, stream)?, sample_rate)
        };
        let seconds = pcm.len() as f64 / sample_rate as f64;
        py.detach(|| ptts::wav::write_wav_file(&path, &pcm, sample_rate as u32).py())?;
        Ok(seconds)
    }

    /// Synthesize `text`, yielding float32 chunks as the decoder produces them.
    ///
    /// The returned object is an iterator; dropping it stops the generation.
    #[pyo3(signature = (text, *, voice=None, temperature=None, seed=None, cfg_coef=None))]
    fn stream(
        &self,
        py: Python<'_>,
        text: &str,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<AudioStream> {
        let opts = options(voice, temperature, seed, cfg_coef);
        let synth = self.lock()?;
        let stream = py.detach(|| synth.stream_with(text, &opts).py())?;
        Ok(AudioStream { inner: Mutex::new(Some(stream)) })
    }

    /// Tokenize `text` the way `synth` would.
    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        self.lock()?.tokenize(text).py()
    }

    /// Synthesize from token ids produced elsewhere, skipping tokenization.
    ///
    /// `frames_after_eos` is the tail length: 3 for a very short prompt, 1
    /// otherwise.
    #[pyo3(signature = (tokens, *, frames_after_eos=1, voice=None, temperature=None, seed=None, cfg_coef=None))]
    #[allow(clippy::too_many_arguments)]
    fn synth_tokens<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<u32>,
        frames_after_eos: usize,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let opts = options(voice, temperature, seed, cfg_coef);
        let stream = self.lock()?.stream_tokens(tokens, frames_after_eos, &opts).py()?;
        let pcm = drain(py, stream)?;
        Ok(PyArray1::from_vec(py, pcm))
    }

    /// Register a voice from a precomputed embedding file, replacing any voice
    /// of the same name.
    fn add_voice(&self, py: Python<'_>, name: &str, path: std::path::PathBuf) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        py.detach(move || {
            let mut synth = inner.lock().map_err(|_| poisoned())?;
            synth.add_voice_file(name, &path).py()
        })
    }

    /// Register a voice from an in-memory conditioning embedding of shape
    /// `[T, dim]` or `[1, T, dim]`.
    ///
    /// `null_embedding` is the encoding of equal-length silence, which CFG
    /// needs on models whose null branch is conditioned on silence.
    #[pyo3(signature = (name, embedding, *, null_embedding=None))]
    fn add_voice_from_embedding(
        &self,
        name: &str,
        embedding: PyReadonlyArrayDyn<'_, f32>,
        null_embedding: Option<PyReadonlyArrayDyn<'_, f32>>,
    ) -> PyResult<()> {
        let (emb, frames, dim) = embedding_dims(&embedding)?;
        let null = match null_embedding.as_ref() {
            None => None,
            Some(arr) => Some(embedding_dims(arr)?.0),
        };
        self.lock()?.add_voice_from_embedding(name, &emb, frames, dim, null.as_deref()).py()
    }

    /// Clone a voice from ~10s of speech, given as float32 PCM at
    /// `voice_prompt_sample_rate`.
    fn clone_voice(
        &self,
        py: Python<'_>,
        name: &str,
        pcm: numpy::PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let pcm = pcm.as_slice()?.to_vec();
        let inner = Arc::clone(&self.inner);
        py.detach(move || {
            let mut synth = inner.lock().map_err(|_| poisoned())?;
            synth.add_voice_from_pcm(name, &pcm).py()
        })
    }

    /// True if this checkpoint can clone voices from audio.
    #[getter]
    fn supports_voice_cloning(&self) -> PyResult<bool> {
        Ok(self.lock()?.supports_voice_cloning())
    }

    fn __repr__(&self) -> PyResult<String> {
        let synth = self.lock()?;
        Ok(format!(
            "TTS(device='{}', quant='{}', sample_rate={}, voices={:?})",
            synth.device_name(),
            synth.quant().as_str(),
            synth.sample_rate(),
            synth.voices()
        ))
    }
}

impl Tts {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Synth>> {
        self.inner.lock().map_err(|_| poisoned())
    }
}

/// A panic inside a `&mut self` method would leave the model half-updated, so a
/// poisoned lock is reported rather than papered over.
fn poisoned() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("the model is unusable: a previous call panicked")
}

fn options(
    voice: Option<String>,
    temperature: Option<f32>,
    seed: Option<u64>,
    cfg_coef: Option<f32>,
) -> SpeechOptions {
    SpeechOptions { voice, temperature, seed, cfg_coef, max_tokens_per_chunk: None }
}

/// Collect a whole stream, letting Ctrl-C through between chunks.
///
/// The GIL is released while waiting on each chunk and reacquired to check for
/// signals, so a long generation stays interruptible without the caller passing
/// a polling interval.
fn drain(py: Python<'_>, stream: SpeechStream) -> PyResult<Vec<f32>> {
    let mut pcm = Vec::new();
    let mut stream = stream;
    loop {
        let next = py.detach(|| stream.next());
        match next {
            None => return Ok(pcm),
            Some(chunk) => pcm.extend_from_slice(&chunk.py()?),
        }
        py.check_signals()?;
    }
}

/// An in-progress generation. Iterate it for float32 chunks.
#[pyclass(module = "ptts")]
struct AudioStream {
    // `SpeechStream` owns an mpsc receiver, which is `Send` but not `Sync`,
    // while `#[pyclass]` wants both; the mutex bridges that. `None` after
    // `close`, so a closed stream iterates as empty rather than raising.
    inner: Mutex<Option<SpeechStream>>,
}

#[pymethods]
impl AudioStream {
    /// Sample rate of the chunks, in Hz.
    #[getter]
    fn sample_rate(&self) -> PyResult<usize> {
        match self.inner.lock().map_err(|_| poisoned())?.as_ref() {
            Some(stream) => Ok(stream.sample_rate()),
            None => Err(pyo3::exceptions::PyValueError::new_err("this stream is closed")),
        }
    }

    /// Stop generating and release the worker threads.
    fn close(&self) -> PyResult<()> {
        *self.inner.lock().map_err(|_| poisoned())? = None;
        Ok(())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let mut guard = slf.inner.lock().map_err(|_| poisoned())?;
        let Some(stream) = guard.as_mut() else { return Ok(None) };
        match py.detach(|| stream.next()) {
            None => Ok(None),
            Some(chunk) => Ok(Some(PyArray1::from_vec(py, chunk.py()?))),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Number of CPU threads used for tensor ops.
#[pyfunction]
fn get_num_threads() -> usize {
    xn::utils::get_num_threads()
}

/// Set the number of CPU threads used for tensor ops. Call before loading a
/// model: it sizes a global thread pool that is built once.
#[pyfunction]
fn set_num_threads(num_threads: usize) {
    xn::utils::set_num_threads(num_threads);
}

/// The device names this build accepts, most capable first. `"auto"` picks the
/// first of these.
#[pyfunction]
fn available_devices() -> Vec<&'static str> {
    let mut devices = vec![];
    if cfg!(feature = "cuda") {
        devices.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        devices.push("vulkan");
    }
    if cfg!(feature = "metal") {
        devices.push("metal");
    }
    devices.push("cpu");
    devices
}

/// The weight formats `quant=` accepts. All are CPU-only.
#[pyfunction]
fn available_quants() -> Vec<&'static str> {
    vec!["f32", "q8_0", "q8_1", "q8k", "q6k", "q5_0", "q5_1", "q5k", "q4_0", "q4_1", "q4k"]
}

/// Report the runtime's build configuration, for bug reports.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    info.set_item("version", env!("CARGO_PKG_VERSION"))?;
    info.set_item("devices", available_devices())?;
    info.set_item("avx", xn::with_avx())?;
    info.set_item("neon", xn::with_neon())?;
    info.set_item("f16c", xn::with_f16c())?;
    info.set_item("threads", xn::utils::get_num_threads())?;
    Ok(info)
}

#[pymodule(name = "_ptts")]
fn ptts_(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Tts>()?;
    m.add_class::<AudioStream>()?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(available_quants, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    Ok(())
}
