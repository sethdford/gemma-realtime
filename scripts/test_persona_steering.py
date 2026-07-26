#!/usr/bin/env python3
"""Tests for persona_steering.py coefficient handling + no-op contract (no model).

The residual-add itself needs the live model (validated by steering_sweep.py); the
testable surface here is set_active's filtering and the installed/active gating
that backs the byte-identical-when-inactive guarantee."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import persona_steering as ps


def _reset(installed, vectors=None):
    ps._STATE.update({"installed": installed, "vectors": vectors or {}, "active": {}})


def _write_vec(path, n_layers, hidden):
    """Write a <trait>.safetensors shaped [n_layers, hidden] under path."""
    import mlx.core as mx
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), {"v": mx.zeros((n_layers, hidden))})


# ── model-namespaced vectors ─────────────────────────────────────────────
# A trait vector is [n_layers, hidden] for ONE architecture. The loader used a
# single flat directory, so when production flipped gemma-4-31b-it-4bit (60
# layers) -> GLM-4.5-Air-4bit (46) on 2026-07-26, the gemma vectors kept being
# loaded and every model load logged `probe FAIL ... (60, 5376) != [46, hidden]`
# while the server ran unsteered. Keyed by base, a flip strands nothing.

def test_model_slug_strips_org_prefix():
    assert ps.model_slug("mlx-community/GLM-4.5-Air-4bit") == "GLM-4.5-Air-4bit"
    assert ps.model_slug("GLM-4.5-Air-4bit") == "GLM-4.5-Air-4bit"
    assert ps.model_slug(None) == ""


def test_vectors_are_loaded_from_the_per_model_dir(tmp=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_vec(root / "GLM-4.5-Air-4bit" / "formality.safetensors", 46, 4096)
        got = ps.load_vectors(root, model_id="mlx-community/GLM-4.5-Air-4bit")
        assert set(got) == {"formality"}, got
        assert tuple(got["formality"].shape) == (46, 4096)


def test_other_models_vectors_are_not_loaded():
    # The bug: gemma vectors served to a GLM process. With a per-model dir
    # present for GLM, gemma's must not be reachable.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_vec(root / "GLM-4.5-Air-4bit" / "formality.safetensors", 46, 4096)
        _write_vec(root / "gemma-4-31b-it-4bit" / "warmth.safetensors", 60, 5376)
        got = ps.load_vectors(root, model_id="mlx-community/GLM-4.5-Air-4bit")
        assert "warmth" not in got, f"leaked another base's vectors: {list(got)}"
        assert tuple(got["formality"].shape) == (46, 4096)


def test_legacy_flat_dir_still_loads():
    # Back-compat: existing installs keep working when no per-model dir exists.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_vec(root / "formality.safetensors", 60, 5376)
        got = ps.load_vectors(root, model_id="mlx-community/gemma-4-31b-it-4bit")
        assert tuple(got["formality"].shape) == (60, 5376)


def test_missing_dir_yields_no_vectors():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        got = ps.load_vectors(Path(d) / "nope", model_id="whatever/x")
        assert got == {}, got


def test_set_active_noop_when_not_installed():
    _reset(False)
    ps.set_active({"formality": 1.0})
    assert ps._STATE["active"] == {}  # nothing armed if not installed


def test_set_active_drops_unknown_traits():
    _reset(True, {"formality": object()})
    ps.set_active({"formality": 0.6, "bogus": 1.0})
    assert ps._STATE["active"] == {"formality": 0.6}


def test_set_active_drops_zero_and_nonfinite():
    _reset(True, {"a": 1, "b": 1, "c": 1, "d": 1})
    ps.set_active({"a": 0.0, "b": float("inf"), "c": float("nan"), "d": 0.7})
    assert ps._STATE["active"] == {"d": 0.7}


def test_set_active_clamps_to_safe_envelope():
    # Phase-2 safe range is [-1, 1] (default); larger magnitudes clamp.
    _reset(True, {"a": 1, "b": 1})
    ps.set_active({"a": 5.0, "b": -3.0})
    assert ps._STATE["active"] == {"a": 1.0, "b": -1.0}


def test_set_active_coerces_numeric_strings():
    _reset(True, {"warmth": 1})
    ps.set_active({"warmth": "0.5"})
    assert ps._STATE["active"] == {"warmth": 0.5}


def test_clear_active():
    _reset(True, {"x": 1})
    ps.set_active({"x": 1.0})
    assert ps._STATE["active"]
    ps.clear_active()
    assert ps._STATE["active"] == {}


def test_empty_active_means_noop_path():
    # The patch returns the original output unchanged when active is empty; this
    # asserts the gating value that drives that branch (no coeffs => no steering).
    _reset(True, {"x": 1})
    ps.set_active({})
    assert ps._STATE["active"] == {}
    ps.set_active(None)
    assert ps._STATE["active"] == {}


def test_layer_band_parsing(monkeypatch_env=None):
    import os
    os.environ["GEMMA_STEER_LAYERS"] = "0.25-0.75"
    assert ps._layer_band(60) == (15, 45)
    os.environ["GEMMA_STEER_LAYERS"] = "all"
    assert ps._layer_band(60) == (0, 60)
    os.environ["GEMMA_STEER_LAYERS"] = "garbage"
    assert ps._layer_band(60) == (0, 60)
    del os.environ["GEMMA_STEER_LAYERS"]


def main():
    tests = [test_set_active_noop_when_not_installed, test_set_active_drops_unknown_traits,
             test_set_active_drops_zero_and_nonfinite, test_set_active_clamps_to_safe_envelope,
             test_set_active_coerces_numeric_strings,
             test_clear_active, test_empty_active_means_noop_path, test_layer_band_parsing,
             test_model_slug_strips_org_prefix,
             test_vectors_are_loaded_from_the_per_model_dir,
             test_other_models_vectors_are_not_loaded,
             test_legacy_flat_dir_still_loads,
             test_missing_dir_yields_no_vectors]
    print("Testing persona_steering.py")
    print("=" * 60)
    p = f = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            p += 1
        except Exception as ex:  # noqa: BLE001
            print(f"✗ {t.__name__}: {type(ex).__name__}: {ex}")
            f += 1
    print("=" * 60)
    print(f"Results: {p} passed, {f} failed")
    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
