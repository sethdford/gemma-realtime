"""LoRA adapter binding tests for scripts/mlx-server.py (2026-09-03).

Background: the serving adapter on :8741 was a silent no-op. The server called
``model.load_weights(<adapter tensors>, strict=False)`` on the BASE model, and
mlx's ``Module.update(strict=False)`` drops every key the module does not
already have — which is every ``lora_a``/``lora_b`` tensor, because no LoRA
layers had been injected. ``tensors_loaded`` was ``len(list)`` so /health said
``adapter_applied=true`` while serving raw base weights.

These tests pin the fix on a tiny model (no real checkpoint, no second MLX
loader — h-uman rule: never two model instances):

  * binding injects LoRA layers and the bound count is the number of adapter
    keys that actually intersect the model's parameters AFTER binding;
  * bound adapter changes the forward output vs base;
  * swapping replaces (does not nest) LoRA layers;
  * an adapter whose keys match nothing reports 0 and does not claim success.

Requires mlx + mlx_lm; skipped where they are not installed.
"""
import importlib.util
import json
import os
import sys

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")
from mlx_lm.models.switch_layers import SwitchLinear  # noqa: E402
from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)

D = 4       # hidden size of the toy model
R = 2       # LoRA rank
N_BLOCKS = 2
N_EXPERTS = 2
N_LORA_MODULES = N_BLOCKS * 2   # each block: one Linear + one SwitchLinear


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "mlx_server_adapter_under_test", os.path.join(_SCRIPTS, "mlx-server.py"))
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


class _Block(nn.Module):
    """One dense projection + one MoE-style SwitchLinear, mirroring the two
    leaf types the production GLM-4.5-Air adapter binds to
    (``self_attn.*_proj`` → LoRALinear, ``mlp.switch_mlp.*`` → LoRASwitchLinear).
    Routing is fixed to expert 0 so the forward is deterministic."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D, D, bias=False)
        self.switch = SwitchLinear(D, D, num_experts=N_EXPERTS, bias=False)

    def __call__(self, x):
        x = self.proj(x)
        idx = mx.zeros((x.shape[0], 1), dtype=mx.int32)
        return self.switch(x[:, None, :], idx).reshape(x.shape)


class TinyModel(nn.Module):
    """Mirrors the mlx_lm model contract that linear_to_lora_layers relies on:
    a ``layers`` list of blocks containing convertible leaves."""

    def __init__(self):
        super().__init__()
        self.layers = [_Block() for _ in range(N_BLOCKS)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _lora_shapes():
    """Key → shape of every LoRA tensor mlx_lm would train on TinyModel — the
    same names ``model.trainable_parameters()`` saves to adapters.safetensors."""
    probe = TinyModel()
    probe.freeze()  # as mlx_lm training does, so only lora_a/lora_b are trainable
    linear_to_lora_layers(probe, N_BLOCKS, {"rank": R, "scale": 2.0, "dropout": 0.0})
    return {k: v.shape for k, v in tree_flatten(probe.trainable_parameters())}


def _write_adapter(dirpath, seed, rename=None):
    """Write adapter_config.json + adapters.safetensors in mlx_lm's layout.
    All tensors are non-zero so a bound adapter MUST move the output.
    ``rename`` maps each real key to a bogus one (for the unbindable case)."""
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, "adapter_config.json"), "w") as f:
        json.dump({
            "fine_tune_type": "lora",
            "num_layers": N_BLOCKS,
            "lora_parameters": {"rank": R, "scale": 2.0, "dropout": 0.0},
        }, f)
    mx.random.seed(seed)
    tensors = {}
    for k, shape in _lora_shapes().items():
        tensors[rename(k) if rename else k] = mx.random.normal(shape) + 0.5
    mx.save_safetensors(os.path.join(dirpath, "adapters.safetensors"), tensors)
    return len(tensors)


def _n_lora_modules(model):
    return sum(1 for _, m in model.named_modules()
               if isinstance(m, (LoRALinear, LoRASwitchLinear)))


@pytest.fixture()
def srv():
    srv = _load_server()
    srv.model = None
    srv.adapter_path_global = None
    srv.tensors_loaded_global = 0
    return srv


@pytest.fixture()
def x():
    mx.random.seed(123)
    return mx.random.normal((1, D))


def test_bind_adapter_injects_lora_and_changes_output(srv, tmp_path, x):
    model = TinyModel()
    y_base = model(x)
    n_keys = _write_adapter(tmp_path / "a", seed=1)

    bound = srv._bind_adapter(model, str(tmp_path / "a"))

    assert bound == n_keys == 2 * N_LORA_MODULES
    assert _n_lora_modules(model) == N_LORA_MODULES
    y_lora = model(x)
    assert not mx.allclose(y_base, y_lora).item(), "bound adapter must change the output"


def test_apply_adapter_weights_sets_honest_global_count(srv, tmp_path):
    srv.model = TinyModel()
    n_keys = _write_adapter(tmp_path / "a", seed=1)

    n = srv._apply_adapter_weights(str(tmp_path / "a"))

    assert n == n_keys
    assert srv.tensors_loaded_global == n_keys


def test_swap_replaces_lora_layers_instead_of_nesting(srv, tmp_path, x):
    srv.model = TinyModel()
    _write_adapter(tmp_path / "a", seed=1)
    _write_adapter(tmp_path / "b", seed=2)

    srv._apply_adapter_weights(str(tmp_path / "a"))
    y_a = srv.model(x)
    srv._apply_adapter_weights(str(tmp_path / "b"))
    y_b = srv.model(x)

    assert _n_lora_modules(srv.model) == N_LORA_MODULES, "swap must not stack LoRA on LoRA"
    assert not mx.allclose(y_a, y_b).item(), "different adapter must give different output"


def test_load_with_adapter_binds_and_reports(srv, tmp_path, x, capsys):
    n_keys = _write_adapter(tmp_path / "a", seed=1)
    base = TinyModel()
    y_base = base(x)

    model, tok = srv._load_with_adapter(lambda name: (base, "tok"), "tiny", str(tmp_path / "a"))

    assert tok == "tok"
    assert srv.tensors_loaded_global == n_keys
    assert not mx.allclose(y_base, model(x)).item()
    assert "Bound" in capsys.readouterr().out


def test_unbindable_adapter_reports_zero_and_warns(srv, tmp_path, x, capsys):
    """Keys that match nothing after injection must NOT be reported as applied.
    This is the exact silent no-op the fix removes — the old code reported
    len(list) here."""
    _write_adapter(tmp_path / "bad", seed=1, rename=lambda k: k.replace("layers.", "nope.", 1))
    base = TinyModel()
    y_base = base(x)

    model, _ = srv._load_with_adapter(lambda name: (base, None), "tiny", str(tmp_path / "bad"))

    assert srv.tensors_loaded_global == 0
    assert mx.allclose(y_base, model(x)).item()
    assert "WARNING" in capsys.readouterr().out


def test_swap_to_unbindable_adapter_raises_so_caller_reverts(srv, tmp_path):
    srv.model = TinyModel()
    _write_adapter(tmp_path / "bad", seed=1, rename=lambda k: k.replace("layers.", "nope.", 1))

    with pytest.raises(RuntimeError, match="0 of"):
        srv._apply_adapter_weights(str(tmp_path / "bad"))
    assert srv.tensors_loaded_global == 0


def test_remove_lora_layers_unwraps_switch_experts_too(srv, tmp_path):
    """mlx_lm 0.31's remove_lora_layers leaves LoRASwitchLinear in place, which
    makes the second bind on a MoE model raise. The server's remover must
    restore every wrapper so the model round-trips to bare leaves."""
    model = TinyModel()
    _write_adapter(tmp_path / "a", seed=1)
    srv._bind_adapter(model, str(tmp_path / "a"))
    assert _n_lora_modules(model) == N_LORA_MODULES

    removed = srv._remove_lora_layers(model)

    assert removed == N_LORA_MODULES
    assert _n_lora_modules(model) == 0
    assert all(isinstance(b.proj, nn.Linear) and isinstance(b.switch, SwitchLinear)
               for b in model.layers)
    assert not any(k.endswith(("lora_a", "lora_b")) for k, _ in tree_flatten(model.parameters()))
