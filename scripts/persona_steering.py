"""Phase 1 — activation steering hook for the local Gemma model.

Adds `alpha * v_l` to the residual stream at each transformer layer to control
voice traits (formality, verbosity, warmth, humor) at inference time. Vectors are
the per-layer directions extracted by extract_persona_vectors.py.

Design contract (CRITICAL):
  - install_steering() class-patches DecoderLayer.__call__ ONCE at model load
    (mlx looks up __call__ on the type, not the instance).
  - When NO steering is active (no coeffs, or all alpha == 0), the patch returns
    the layer's original output UNTOUCHED — byte-identical to unsteered. This is
    the backward-compat / default-off guarantee.
  - Coefficients are set per-request via set_active()/clear_active(); the server
    holds model_lock during a generation, so a module-global active dict is safe.

Capability probe: if vectors are missing or their hidden dim doesn't match the
model, install_steering() declines and the server runs unsteered (MLX-churn /
config-drift safety — steering must never break the generation path).
"""
import os
from pathlib import Path

_STATE = {
    "installed": False,
    "vectors": {},      # trait -> mx.array [n_layers, hidden]
    "active": {},       # trait -> float alpha (per request)
    "lo": 0,            # inclusive layer band
    "hi": 10 ** 9,      # exclusive layer band
    "n_layers": 0,
}


def get_layers(model):
    """Find the transformer layer list across mlx_lm wrapper shapes."""
    for path in (("model", "layers"), ("language_model", "model", "layers"),
                 ("model", "model", "layers"), ("layers",)):
        obj = model
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and obj is not None:
            return obj
    raise RuntimeError("could not locate transformer layers on model")


def load_vectors(out_dir):
    """Load {trait: mx.array[n_layers, hidden]} from <out_dir>/<trait>.safetensors.
    Missing dir or files -> {} (steering simply stays off)."""
    import mlx.core as mx
    out_dir = Path(out_dir)
    vectors = {}
    if not out_dir.is_dir():
        return vectors
    for f in sorted(out_dir.glob("*.safetensors")):
        try:
            d = mx.load(str(f))
            if "v" in d:
                vectors[f.stem] = d["v"]
        except Exception as ex:  # noqa: BLE001
            print(f"[steering] skip {f.name}: {ex}", flush=True)
    return vectors


def _layer_band(n_layers):
    """Resolve the steered layer band from GEMMA_STEER_LAYERS ('all' or 'lo-hi'
    fractions, e.g. '0.3-0.7'). Default: all layers."""
    raw = os.environ.get("GEMMA_STEER_LAYERS", "all").strip().lower()
    if raw in ("", "all"):
        return 0, n_layers
    try:
        lo_s, hi_s = raw.split("-")
        lo = max(0, int(float(lo_s) * n_layers))
        hi = min(n_layers, int(float(hi_s) * n_layers))
        return (lo, hi) if lo < hi else (0, n_layers)
    except Exception:  # noqa: BLE001
        return 0, n_layers


def install_steering(layers, vectors):
    """Class-patch the layer type to apply active steering to its residual output.
    Returns True if installed (probe passed), False if declined."""
    import mlx.core as mx

    if not layers or not vectors:
        return False
    n_layers = len(layers)

    # Capability probe: every vector must be [n_layers, hidden] matching the model.
    hidden = None
    for trait, v in vectors.items():
        if v.ndim != 2 or v.shape[0] != n_layers:
            print(f"[steering] probe FAIL: {trait} shape {tuple(v.shape)} != "
                  f"[{n_layers}, hidden]; running unsteered", flush=True)
            return False
        hidden = v.shape[1] if hidden is None else hidden
        if v.shape[1] != hidden:
            print("[steering] probe FAIL: inconsistent hidden dims; unsteered", flush=True)
            return False

    lo, hi = _layer_band(n_layers)
    _STATE.update({"vectors": vectors, "n_layers": n_layers, "lo": lo, "hi": hi})

    layer_cls = type(layers[0])
    orig_call = layer_cls.__call__
    for i, layer in enumerate(layers):
        layer._steer_idx = i

    def patched(self, x, *a, **k):
        out = orig_call(self, x, *a, **k)
        active = _STATE["active"]
        if not active:
            return out  # NO-OP: byte-identical to unsteered (the contract)
        idx = getattr(self, "_steer_idx", None)
        if idx is None or not (_STATE["lo"] <= idx < _STATE["hi"]):
            return out
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        delta = None
        for trait, alpha in active.items():
            if alpha == 0:
                continue
            v = _STATE["vectors"].get(trait)
            if v is None:
                continue
            term = alpha * v[idx]  # [hidden] broadcasts over [batch, seq, hidden]
            delta = term if delta is None else delta + term
        if delta is None:
            return out
        h = h + delta
        return (h,) + tuple(out[1:]) if is_tuple else h

    layer_cls.__call__ = patched
    _STATE["installed"] = True
    print(f"[steering] installed: {len(vectors)} traits {list(vectors)}, "
          f"layers [{lo},{hi}) of {n_layers}", flush=True)
    return True


def _max_alpha():
    """Safety clamp on |alpha|. The Phase-2 sweep (steering_sweep.py) found
    capability stays intact (3/3 probes) within alpha in [-1, 1] for every trait
    and degrades at |alpha|=2 (runaway length / broken instruction-following) —
    the lora-scale-over-amplification failure mode. So clamp to 1.0 by default.
    Tunable via GEMMA_STEER_MAX_ALPHA for experimentation."""
    raw = os.environ.get("GEMMA_STEER_MAX_ALPHA", "").strip()
    if not raw:
        return 1.0
    try:
        v = float(raw)
        return abs(v) if v == v and abs(v) != float("inf") and v != 0 else 1.0
    except ValueError:
        return 1.0


def set_active(coeffs):
    """Arm steering for the current request. coeffs: {trait: alpha}. Unknown
    traits / non-finite / zero alphas are dropped; |alpha| is clamped to the
    measured safe envelope (_max_alpha). No-op if steering not installed."""
    if not _STATE["installed"] or not isinstance(coeffs, dict):
        _STATE["active"] = {}
        return
    cap = _max_alpha()
    clean = {}
    for trait, alpha in coeffs.items():
        if trait not in _STATE["vectors"]:
            continue
        try:
            a = float(alpha)
        except (TypeError, ValueError):
            continue
        if a != a or abs(a) == float("inf") or a == 0.0:  # drop nan/inf/zero
            continue
        # Clamp to the capability-safe range measured in Phase 2.
        a = max(-cap, min(cap, a))
        clean[trait] = a
    _STATE["active"] = clean


def clear_active():
    _STATE["active"] = {}


def is_installed():
    return _STATE["installed"]
