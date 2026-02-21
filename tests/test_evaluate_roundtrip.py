from __future__ import annotations

import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from emotionbridge.scripts import evaluate_roundtrip as roundtrip


def _make_wave(
    *,
    freq: float,
    sr: int = 24000,
    seconds: float = 1.2,
) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float64)
    waveform = 0.2 * np.sin(2.0 * np.pi * freq * t)
    waveform += 0.03 * np.sin(2.0 * np.pi * (freq * 2.0) * t)
    return waveform.astype(np.float64)


class TestEvaluateRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_wav(self, filename: str, wav: np.ndarray, sr: int = 24000) -> Path:
        path = self.tmp_path / filename
        sf.write(path, wav, sr, format="WAV", subtype="PCM_16")
        return path

    def _write_manifest(self, filename: str, rows: list[dict[str, object]]) -> Path:
        path = self.tmp_path / filename
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def test_manifest_missing_required_key_raises(self) -> None:
        wav_path = self._write_wav("a.wav", _make_wave(freq=220.0))
        manifest_path = self._write_manifest(
            "missing_required.json",
            [{"text": "hello", "audio_path": str(wav_path)}],
        )
        try:
            roundtrip._load_manifest_entries(manifest_path)
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for missing required manifest key"
            raise AssertionError(msg)

    def test_manifest_duplicate_key_raises(self) -> None:
        wav_path = self._write_wav("dup.wav", _make_wave(freq=220.0))
        manifest_path = self._write_manifest(
            "duplicate.json",
            [
                {
                    "target_emotion": "anger",
                    "text": "same text",
                    "audio_path": str(wav_path),
                },
                {
                    "target_emotion": "anger",
                    "text": "same text",
                    "audio_path": str(wav_path),
                },
            ],
        )
        try:
            roundtrip._load_manifest_entries(manifest_path)
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for duplicate manifest key"
            raise AssertionError(msg)

    def test_manifest_pair_mismatch_raises(self) -> None:
        wav_a = self._write_wav("a.wav", _make_wave(freq=220.0))
        wav_b = self._write_wav("b.wav", _make_wave(freq=260.0))

        baseline_manifest = self._write_manifest(
            "baseline.json",
            [
                {
                    "target_emotion": "anger",
                    "text": "text A",
                    "audio_path": str(wav_a),
                },
            ],
        )
        candidate_manifest = self._write_manifest(
            "candidate.json",
            [
                {
                    "target_emotion": "happy",
                    "text": "text B",
                    "audio_path": str(wav_b),
                },
            ],
        )

        baseline_entries = roundtrip._load_manifest_entries(baseline_manifest)
        candidate_entries = roundtrip._load_manifest_entries(candidate_manifest)
        try:
            roundtrip._pair_manifest_entries(baseline_entries, candidate_entries)
        except ValueError:
            pass
        else:
            msg = "Expected ValueError for manifest key mismatch"
            raise AssertionError(msg)

    def test_world_metrics_increase_for_modified_wave(self) -> None:
        ref_path = self._write_wav("ref.wav", _make_wave(freq=220.0))
        same_path = self._write_wav("same.wav", _make_wave(freq=220.0))
        mod_path = self._write_wav("mod.wav", _make_wave(freq=260.0))

        ref, ref_sr = roundtrip._read_audio_mono(ref_path)
        same, same_sr = roundtrip._read_audio_mono(same_path)
        mod, mod_sr = roundtrip._read_audio_mono(mod_path)

        ref_same, deg_same, sr_same = roundtrip._align_pair(ref, ref_sr, same, same_sr)
        ref_mod, deg_mod, sr_mod = roundtrip._align_pair(ref, ref_sr, mod, mod_sr)

        metrics_same = roundtrip._compute_world_metrics(ref_same, deg_same, sr_same)
        metrics_mod = roundtrip._compute_world_metrics(ref_mod, deg_mod, sr_mod)

        assert float(metrics_same["mcd_db"]) <= float(metrics_mod["mcd_db"])
        if np.isfinite(float(metrics_same["f0_rmse_hz"])) and np.isfinite(
            float(metrics_mod["f0_rmse_hz"]),
        ):
            assert float(metrics_same["f0_rmse_hz"]) <= float(metrics_mod["f0_rmse_hz"])

    def test_missing_pesq_dependency_raises_runtime_error(self) -> None:
        original_import = builtins.__import__

        def _fake_import(
            name: str,
            globals_: dict[str, object] | None = None,
            locals_: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ):
            if name == "pesq":
                msg = "No module named 'pesq'"
                raise ModuleNotFoundError(msg)
            return original_import(name, globals_, locals_, fromlist, level)

        with patch("builtins.__import__", side_effect=_fake_import):
            try:
                roundtrip._get_pesq_function()
            except RuntimeError:
                pass
            else:
                msg = "Expected RuntimeError when pesq import fails"
                raise AssertionError(msg)

    def test_go_no_go_decision_logic(self) -> None:
        go = roundtrip._judge_go_no_go(
            pesq_mean=3.8,
            mcd_mean_db=5.5,
            f0_rmse_mean_hz=4.0,
        )
        assert go["label"] == "Go"
        assert go["pass"]

        no_go = roundtrip._judge_go_no_go(
            pesq_mean=3.2,
            mcd_mean_db=6.8,
            f0_rmse_mean_hz=7.2,
        )
        assert no_go["label"] == "No-Go"
        assert not no_go["pass"]
        assert no_go["failure_reasons"]


if __name__ == "__main__":
    unittest.main()
