//! Exercises everything about `Synth` that does not need model weights.
//!
//! The generation path itself needs a checkpoint, so it is covered by the
//! examples rather than here. What is testable without one is the surface most
//! likely to break for a first-time user: argument parsing, feature gating, and
//! the error messages on the paths they will hit by accident.

use ptts::checkpoint::ModelSource;
use ptts::synth::{DeviceKind, Quant, SpeechOptions, Synth};

#[test]
fn quant_parses_every_spelling_the_clis_accept() {
    let cases = [
        ("f32", Quant::F32),
        ("none", Quant::F32),
        ("q8", Quant::Q80),
        ("q8_0", Quant::Q80),
        ("q8_1", Quant::Q81),
        ("q8k", Quant::Q8k),
        ("q6k", Quant::Q6k),
        ("q5", Quant::Q50),
        ("q5_0", Quant::Q50),
        ("q5_1", Quant::Q51),
        ("q5k", Quant::Q5k),
        ("q4", Quant::Q40),
        ("q4_0", Quant::Q40),
        ("q4_1", Quant::Q41),
        ("q4k", Quant::Q4k),
    ];
    for (name, expected) in cases {
        assert_eq!(Quant::parse(name).unwrap(), expected, "parsing {name}");
    }
}

#[test]
fn quant_round_trips_through_its_canonical_name() {
    for quant in [
        Quant::F32,
        Quant::Q80,
        Quant::Q81,
        Quant::Q8k,
        Quant::Q6k,
        Quant::Q50,
        Quant::Q51,
        Quant::Q5k,
        Quant::Q40,
        Quant::Q41,
        Quant::Q4k,
    ] {
        assert_eq!(Quant::parse(quant.as_str()).unwrap(), quant);
    }
}

#[test]
fn unknown_quant_lists_the_valid_ones() {
    let err = Quant::parse("q3k").unwrap_err().to_string();
    assert!(err.contains("q3k"), "{err}");
    assert!(err.contains("q4k"), "should list the supported formats: {err}");
}

#[test]
fn device_parses_and_rejects_unknown_names() {
    assert_eq!(DeviceKind::parse("auto").unwrap(), DeviceKind::Auto);
    assert_eq!(DeviceKind::parse("cpu").unwrap(), DeviceKind::Cpu);
    assert_eq!(DeviceKind::parse("cuda").unwrap(), DeviceKind::Cuda);
    assert_eq!(DeviceKind::parse("vulkan").unwrap(), DeviceKind::Vulkan);
    assert_eq!(DeviceKind::parse("metal").unwrap(), DeviceKind::Metal);
    let err = DeviceKind::parse("tpu").unwrap_err().to_string();
    assert!(err.contains("tpu"), "{err}");
}

#[test]
fn auto_resolves_to_something_concrete() {
    let resolved = DeviceKind::Auto.resolve();
    assert_ne!(resolved, DeviceKind::Auto);
    // With no GPU feature enabled, `auto` must land on the CPU.
    if !cfg!(any(feature = "cuda", feature = "vulkan", feature = "metal")) {
        assert_eq!(resolved, DeviceKind::Cpu);
    }
}

#[test]
fn explicit_device_survives_resolution() {
    for device in [DeviceKind::Cpu, DeviceKind::Cuda, DeviceKind::Vulkan, DeviceKind::Metal] {
        assert_eq!(device.resolve(), device);
    }
}

#[test]
fn quantization_on_a_gpu_is_rejected_before_any_download() {
    let err = Synth::builder()
        .device(DeviceKind::Cuda)
        .quant(Quant::Q40)
        .build()
        .unwrap_err()
        .to_string();
    assert!(err.contains("CPU-only"), "{err}");
    assert!(err.contains("q4_0"), "the error should name the format: {err}");
}

#[test]
fn a_missing_directory_names_itself() {
    let err = Synth::from_dir("/definitely/not/a/model/dir").unwrap_err().to_string();
    assert!(err.contains("/definitely/not/a/model/dir"), "{err}");
}

#[test]
fn an_empty_directory_lists_the_weight_files_it_looked_for() {
    let dir = std::env::temp_dir().join("ptts-synth-api-empty-dir");
    std::fs::create_dir_all(&dir).unwrap();
    let err = Synth::from_dir(&dir).unwrap_err().to_string();
    assert!(err.contains("model.safetensors"), "should list the candidates: {err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_explicit_weights_name_the_path() {
    let source = ModelSource::Files {
        config: None,
        weights: "/nope/weights.safetensors".into(),
        tokenizer: None,
    };
    let err = source.resolve(0.7).unwrap_err().to_string();
    assert!(err.contains("/nope/weights.safetensors"), "{err}");
}

#[test]
fn an_unparseable_config_names_the_file() {
    let path = std::env::temp_dir().join("ptts-synth-api-bad-config.json");
    std::fs::write(&path, "{ not json").unwrap();
    let source =
        ModelSource::Files { config: Some(path.clone()), weights: "/nope".into(), tokenizer: None };
    let err = source.resolve(0.7).unwrap_err().to_string();
    assert!(err.contains("ptts-synth-api-bad-config.json"), "{err}");
    std::fs::remove_file(&path).ok();
}

#[cfg(not(feature = "hub"))]
#[test]
fn the_hub_path_says_which_feature_is_missing() {
    let err = Synth::from_hub().unwrap_err().to_string();
    assert!(err.contains("hub"), "{err}");
}

#[test]
fn speech_options_build_up_fluently() {
    let opts = SpeechOptions::default()
        .voice("alba")
        .temperature(0.5)
        .seed(7)
        .cfg_coef(1.5)
        .max_tokens_per_chunk(32);
    assert_eq!(opts.voice.as_deref(), Some("alba"));
    assert_eq!(opts.temperature, Some(0.5));
    assert_eq!(opts.seed, Some(7));
    assert_eq!(opts.cfg_coef, Some(1.5));
    assert_eq!(opts.max_tokens_per_chunk, Some(32));
}

#[test]
fn speech_options_default_to_the_builders_settings() {
    let opts = SpeechOptions::default();
    assert!(opts.voice.is_none());
    assert!(opts.temperature.is_none());
    assert!(opts.seed.is_none());
    assert!(opts.cfg_coef.is_none());
}
