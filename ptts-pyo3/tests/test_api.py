"""Tests for the `ptts` Python API that do not need model weights.

The published checkpoint is gated, so loading a model is covered by
`test_model.py`, which skips unless `PTTS_MODEL_DIR` points at a local one.
What is here is the surface a first-time user touches: the module's exports,
the type stubs matching the implementation, and the errors produced by wrong
arguments.

    pip install -e '.[test]' && pytest
"""

import inspect
import re
from pathlib import Path

import pytest

import ptts


def test_version_is_exported_and_sane():
    assert re.fullmatch(r"\d+\.\d+\.\d+", ptts.__version__), ptts.__version__


def test_public_exports():
    exported = {n for n in dir(ptts) if not n.startswith("_")}
    assert {
        "TTS",
        "AudioStream",
        "available_devices",
        "available_quants",
        "build_info",
        "get_num_threads",
        "set_num_threads",
    } <= exported


def test_cpu_is_always_an_available_device():
    devices = ptts.available_devices()
    assert devices[-1] == "cpu", "cpu is the fallback, so it comes last"
    assert len(set(devices)) == len(devices)


def test_quants_are_listed_and_include_the_common_ones():
    quants = ptts.available_quants()
    assert "f32" in quants
    assert "q8_0" in quants
    assert "q4k" in quants


def test_build_info_reports_what_a_bug_report_needs():
    info = ptts.build_info()
    assert info["version"] == ptts.__version__
    assert info["devices"] == ptts.available_devices()
    for key in ("avx", "neon", "f16c"):
        assert isinstance(info[key], bool)
    assert isinstance(info["threads"], int)


def test_thread_count_round_trips():
    before = ptts.get_num_threads()
    try:
        ptts.set_num_threads(2)
        assert ptts.get_num_threads() == 2
    finally:
        ptts.set_num_threads(before)


class TestBadArguments:
    """Every one of these must fail before anything is downloaded."""

    def test_unknown_device(self):
        with pytest.raises(ValueError, match="unknown device 'tpu'"):
            ptts.TTS(device="tpu")

    def test_unknown_quant(self):
        with pytest.raises(ValueError, match="unsupported quantization 'q3k'"):
            ptts.TTS(quant="q3k")

    def test_quantization_on_a_gpu_is_rejected(self):
        with pytest.raises(ValueError, match="CPU-only"):
            ptts.TTS(device="cuda", quant="q4k")

    def test_missing_directory_names_itself(self):
        with pytest.raises(ValueError, match="/definitely/not/here"):
            ptts.TTS(dir="/definitely/not/here")

    def test_empty_directory_lists_what_it_wanted(self, tmp_path):
        with pytest.raises(ValueError, match="model.safetensors"):
            ptts.TTS(dir=str(tmp_path))


class TestTypeStubs:
    """The stubs ship in the wheel, so they must describe the real thing."""

    @staticmethod
    def stub_text() -> str:
        for candidate in (
            Path(ptts.__file__).with_name("__init__.pyi"),
            Path(__file__).parent.parent / "ptts.pyi",
        ):
            if candidate.is_file():
                return candidate.read_text()
        pytest.skip("no type stubs found next to the module or in the source tree")

    def test_stubs_are_installed_next_to_the_module(self):
        assert Path(ptts.__file__).with_name("py.typed").is_file()
        assert Path(ptts.__file__).with_name("__init__.pyi").is_file()

    def test_stubs_declare_every_public_name(self):
        stub = self.stub_text()
        for name in dir(ptts):
            if name.startswith("_") and name != "__version__":
                continue
            if name == "ptts":  # the inner extension module, not public API
                continue
            assert name in stub, f"{name} is exported but missing from ptts.pyi"

    def test_stubs_declare_every_tts_member(self):
        stub = self.stub_text()
        for name in dir(ptts.TTS):
            if name.startswith("_"):
                continue
            assert name in stub, f"TTS.{name} is missing from ptts.pyi"

    def test_stubs_declare_every_stream_member(self):
        stub = self.stub_text()
        for name in dir(ptts.AudioStream):
            if name.startswith("_"):
                continue
            assert name in stub, f"AudioStream.{name} is missing from ptts.pyi"

    def test_stubs_are_valid_python(self):
        compile(self.stub_text(), "ptts.pyi", "exec")


class TestKeywordOnlyOptions:
    """Per-call overrides are keyword-only, so they cannot be passed by accident."""

    @pytest.mark.parametrize("method", ["synth", "save", "stream", "synth_tokens"])
    def test_options_are_keyword_only_in_the_stubs(self, method):
        stub = TestTypeStubs.stub_text()
        body = stub.split(f"def {method}(")[1]
        signature = body.split(")")[0]
        assert "*," in signature, f"{method} should take its options keyword-only"

    def test_docstrings_survive_into_python(self):
        # A binding with no docstrings is unusable from a REPL.
        assert ptts.TTS.__doc__
        assert inspect.getdoc(ptts.TTS.synth)
        assert inspect.getdoc(ptts.available_devices)
