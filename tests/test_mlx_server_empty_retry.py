"""Empty-reply local retry + per-request adapter disable (2026-10-01).

Background: since 3c4a582 (2026-09-03) the persona LoRA is actually bound. It
was trained without GLM's chat template, so on prompts ending
``<|assistant|>\\n<think></think>`` it often emits EOS immediately (1 token) or
``</think>`` + EOS (2-3 tokens) — ~13% of requests since 09-19. The
strip-fallback "salvaged" the literal ``</think>`` and returned it as the
reply; the h-uman client stripped that to empty and failed over to cloud.

Pinned here, model-free (a tiny LoRA-bound toy module + a mocked
``mlx_lm.stream_generate``):

  * the empty detector (cases);
  * the salvage / guard never returns pure scaffold such as ``</think>``;
  * an empty first generation is regenerated ONCE with the adapter's LoRA
    scale zeroed, the adapter is restored afterwards (also on exceptions), and
    the retry never touches the cross-turn prompt cache;
  * ``MLX_EMPTY_RETRY=0`` disables the retry;
  * ``"disable_adapter": true`` routes one request to base weights and the
    next request has the adapter again.
"""
import importlib.util
import os
import sys
import time
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
mlx_lm = pytest.importorskip("mlx_lm")
from mlx_lm.tuner.lora import LoRALinear  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)

LORA_SCALE = 2.0


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "mlx_server_empty_retry_under_test", os.path.join(_SCRIPTS, "mlx-server.py"))
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def __call__(self, x):
        return self.proj(x)


class TinyLoRAModel(nn.Module):
    """A ``layers`` list with LoRA injected, the way _bind_adapter leaves the
    production model."""

    def __init__(self):
        super().__init__()
        self.layers = [_Block() for _ in range(2)]
        linear_to_lora_layers(self, 2, {"rank": 2, "scale": LORA_SCALE, "dropout": 0.0})

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _scales(model):
    return [m.scale for _, m in model.named_modules() if isinstance(m, LoRALinear)]


class FakeGenerate:
    """Stands in for mlx_lm.stream_generate. Each call pops the next scripted
    (text, generation_tokens) and records the LoRA scales and whether a
    cross-turn prompt cache was passed AT CALL TIME."""

    def __init__(self, model, outputs, raise_on_call=None):
        self.model = model
        self.outputs = list(outputs)
        self.calls = []
        self.raise_on_call = raise_on_call

    def __call__(self, model, tokenizer, prompt=None, max_tokens=256, **kw):
        self.calls.append({"scales": _scales(self.model),
                           "prompt_cache": kw.get("prompt_cache")})
        if self.raise_on_call is not None and len(self.calls) == self.raise_on_call:
            raise RuntimeError("boom")
        text, ntok = self.outputs.pop(0)
        yield SimpleNamespace(text=text, prompt_tokens=11, generation_tokens=ntok)


@pytest.fixture()
def srv(monkeypatch):
    srv = _load_server()
    srv.model = TinyLoRAModel()
    srv.processor = None
    srv.use_lm_path = True
    srv.adapter_path_global = "/x/adapters/seth-glm-air-test"
    srv.tensors_loaded_global = 8
    srv.turbo_cache = None
    srv.lm_prompt_cache = None
    monkeypatch.setattr(srv, "prepare_prompt_lm", lambda messages, skip=None: "PROMPT")
    monkeypatch.delenv("MLX_EMPTY_RETRY", raising=False)
    return srv


MSGS = [{"role": "user", "content": "hey you around"}]


def _install(monkeypatch, fake):
    monkeypatch.setattr(mlx_lm, "stream_generate", fake)


# --------------------------------------------------------------------------
# Empty detector
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,visible,ntok,expected", [
    ("", "", 1, True),                                  # immediate EOS
    ("</think>", "", 2, True),                          # closer + EOS
    ("</think>", "</think>", 3, True),                  # the salvaged-scaffold shape
    ("  \n</think>\n ", "", 3, True),                   # whitespace around scaffold
    ("<think></think>", "", 4, True),                   # empty pair, >3 tokens
    ("<|assistant|>", "", 2, True),                     # other GLM scaffold
    ("<|channel>thought<channel|>", "", 5, True),       # gemma scaffold
    ("ok", "", 1, True),                                # <=3 toks, visible empty (guarded)
    ("yeah", "yeah", 1, False),                         # short real reply
    ("</think>yeah im around", "yeah im around", 6, False),
    ("I should reply casually and briefly here", "", 40, False),  # long, guarded: not retried
])
def test_empty_detector(srv, raw, visible, ntok, expected):
    assert srv._is_empty_generation(raw, visible, ntok) is expected


@pytest.mark.parametrize("text,expected", [
    ("", True), ("   ", True), ("</think>", True), ("<think>", True),
    ("</think>\n<|assistant|>", True), ("<end_of_turn>", True), ("<eos>", True),
    ("<|channel>thought", True), ("<channel|>", True),
    ("hey", False), ("</think>hey", False), ("<3", False), ("a <b> c", False),
])
def test_pure_scaffold(srv, text, expected):
    assert srv._is_pure_scaffold(text) is expected


# --------------------------------------------------------------------------
# Salvage never returns scaffold
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["</think>", "</think>\n", "<think></think>",
                                 "</think>\n</think>", "<|assistant|>"])
def test_finalize_never_returns_bare_scaffold(srv, raw):
    clean, _runaway = srv.finalize_generation(raw, MSGS)
    assert clean == ""


def test_finalize_still_salvages_real_text(srv):
    clean, runaway = srv.finalize_generation("</think>yeah im around", MSGS)
    assert (clean, runaway) == ("yeah im around", False)


# --------------------------------------------------------------------------
# Retry path
# --------------------------------------------------------------------------

def test_empty_reply_retried_once_without_adapter_and_adapter_restored(srv, monkeypatch, capsys):
    fake = FakeGenerate(srv.model, [("</think>", 2), ("yeah im around", 5)])
    _install(monkeypatch, fake)
    assert _scales(srv.model) == [LORA_SCALE, LORA_SCALE]

    text, _pt, gen = srv.generate_response(MSGS, 512, 0.7)

    assert text == "yeah im around"
    assert len(fake.calls) == 2, "exactly one retry"
    assert fake.calls[0]["scales"] == [LORA_SCALE, LORA_SCALE]   # first: adapter on
    assert fake.calls[1]["scales"] == [0.0, 0.0]                 # retry: adapter off
    assert fake.calls[1]["prompt_cache"] is None                 # retry: no cross-turn cache
    assert _scales(srv.model) == [LORA_SCALE, LORA_SCALE]        # restored
    assert gen == 2 + 5
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if "[empty-retry]" in l]
    assert line == ["  [empty-retry] adapter=seth-glm-air-test first_tokens=2 retry_tokens=5 ok=True"]
    assert "yeah im around" not in line[0]  # never log output text


def test_retry_also_empty_returns_empty_string_not_scaffold(srv, monkeypatch, capsys):
    fake = FakeGenerate(srv.model, [("</think>", 3), ("</think>", 2)])
    _install(monkeypatch, fake)

    text, _pt, _gen = srv.generate_response(MSGS, 512, 0.7)

    assert text == ""
    assert len(fake.calls) == 2
    assert _scales(srv.model) == [LORA_SCALE, LORA_SCALE]
    assert "ok=False" in capsys.readouterr().out


def test_adapter_restored_when_retry_raises(srv, monkeypatch, capsys):
    fake = FakeGenerate(srv.model, [("", 1)], raise_on_call=2)
    _install(monkeypatch, fake)

    text, _pt, _gen = srv.generate_response(MSGS, 512, 0.7)

    assert text == ""
    assert _scales(srv.model) == [LORA_SCALE, LORA_SCALE]
    assert srv._base_weights_depth == 0
    out = capsys.readouterr().out
    assert "[empty-retry]" in out and "ok=False" in out and "error=RuntimeError" in out


def test_non_empty_reply_not_retried(srv, monkeypatch, capsys):
    fake = FakeGenerate(srv.model, [("yeah", 1)])
    _install(monkeypatch, fake)

    text, _pt, _gen = srv.generate_response(MSGS, 512, 0.7)

    assert text == "yeah"
    assert len(fake.calls) == 1
    assert "[empty-retry]" not in capsys.readouterr().out


def test_no_retry_when_no_adapter_bound(srv, monkeypatch):
    srv.tensors_loaded_global = 0
    fake = FakeGenerate(srv.model, [("</think>", 2)])
    _install(monkeypatch, fake)

    text, _pt, _gen = srv.generate_response(MSGS, 512, 0.7)

    assert text == ""
    assert len(fake.calls) == 1


@pytest.mark.parametrize("val", ["0", "false", "off", "no"])
def test_kill_switch_disables_retry(srv, monkeypatch, capsys, val):
    monkeypatch.setenv("MLX_EMPTY_RETRY", val)
    fake = FakeGenerate(srv.model, [("</think>", 2)])
    _install(monkeypatch, fake)

    text, _pt, _gen = srv.generate_response(MSGS, 512, 0.7)

    assert text == ""            # still never the bare scaffold
    assert len(fake.calls) == 1  # no retry
    assert "[empty-retry]" not in capsys.readouterr().out


def test_retry_enabled_by_default(srv):
    assert srv._empty_retry_enabled() is True


def test_base_weights_bypass_leaves_cross_turn_cache_untouched(srv):
    """While on base weights, the KV/LCP bookkeeping must not be read or
    written — base-weight KV in the shared cache would leak into the next
    adapter request."""
    sentinel = [object()]
    srv.turbo_cache = sentinel
    srv._cached_system_hash = 12345
    srv._cached_system_ntokens = 77
    with srv._base_weights_request():
        assert srv._kv_kwargs() == {}
        assert srv._plan_cache_for_prompt(MSGS, "PROMPT") == "PROMPT"
        srv._prepare_cache_for_request(MSGS)
        srv._after_prefill(MSGS)
    assert srv._cached_system_hash == 12345
    assert srv._cached_system_ntokens == 77
    assert srv._kv_kwargs()["prompt_cache"] is sentinel


def test_dora_adapter_refuses_toggle_without_mutating(srv):
    class DoRAFake(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 2.0

    srv.model.layers[0].extra = DoRAFake()
    with pytest.raises(srv._AdapterToggleUnsupported):
        with srv._base_weights_request():
            pass
    assert _scales(srv.model) == [LORA_SCALE, LORA_SCALE]
    assert srv._base_weights_depth == 0


# --------------------------------------------------------------------------
# Per-request "disable_adapter"
# --------------------------------------------------------------------------

def _handler(srv, sent):
    h = srv.ChatHandler.__new__(srv.ChatHandler)
    h._send_json = lambda code, obj, headers=None: sent.append((code, obj))
    return h


def test_disable_adapter_flag_routes_one_request_to_base_weights(srv, monkeypatch, capsys):
    fake = FakeGenerate(srv.model, [("base reply", 3), ("persona reply", 4)])
    _install(monkeypatch, fake)
    sent = []
    h = _handler(srv, sent)

    h._do_chat_completion({"messages": MSGS, "disable_adapter": True}, "id1", time.time())
    h._do_chat_completion({"messages": MSGS}, "id2", time.time())

    assert fake.calls[0]["scales"] == [0.0, 0.0]
    assert fake.calls[0]["prompt_cache"] is None
    assert fake.calls[1]["scales"] == [LORA_SCALE, LORA_SCALE]  # next request: adapter again
    assert _scales(srv.model) == [LORA_SCALE, LORA_SCALE]
    assert [obj["choices"][0]["message"]["content"] for _c, obj in sent] == \
        ["base reply", "persona reply"]
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "[disable-adapter]" in l]
    assert lines == ["  [disable-adapter] adapter=seth-glm-air-test lora_modules=2"]


def test_disable_adapter_default_false_unchanged(srv, monkeypatch):
    fake = FakeGenerate(srv.model, [("persona reply", 4)])
    _install(monkeypatch, fake)
    sent = []
    _handler(srv, sent)._do_chat_completion({"messages": MSGS}, "id", time.time())
    assert fake.calls[0]["scales"] == [LORA_SCALE, LORA_SCALE]


def test_disable_adapter_restores_adapter_on_exception(srv, monkeypatch):
    fake = FakeGenerate(srv.model, [], raise_on_call=1)
    _install(monkeypatch, fake)
    sent = []
    with pytest.raises(RuntimeError):
        _handler(srv, sent)._do_chat_completion(
            {"messages": MSGS, "disable_adapter": True}, "id", time.time())
    assert _scales(srv.model) == [LORA_SCALE, LORA_SCALE]
    assert srv._base_weights_depth == 0


def test_disable_adapter_skips_empty_retry(srv, monkeypatch):
    """On base weights there is no adapter to blame — no second generation."""
    fake = FakeGenerate(srv.model, [("</think>", 2)])
    _install(monkeypatch, fake)
    sent = []
    _handler(srv, sent)._do_chat_completion(
        {"messages": MSGS, "disable_adapter": True}, "id", time.time())
    assert len(fake.calls) == 1
    assert sent[0][1]["choices"][0]["message"]["content"] == ""


def test_buffered_stream_path_retries_empty_reply(srv, monkeypatch):
    import io
    import json as _json
    fake = FakeGenerate(srv.model, [("</think>", 2), ("yeah im around", 5)])
    _install(monkeypatch, fake)
    h = srv.ChatHandler.__new__(srv.ChatHandler)
    h.wfile = io.BytesIO()

    h._handle_stream_buffered(MSGS, "id", time.time(), 64, 0.7)

    events = [l[len("data: "):] for l in h.wfile.getvalue().decode().splitlines()
              if l.startswith("data: ") and l != "data: [DONE]"]
    chunks = [_json.loads(e) for e in events]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == "yeah im around"
    assert fake.calls[1]["scales"] == [0.0, 0.0]
    assert _scales(srv.model) == [LORA_SCALE, LORA_SCALE]
    assert chunks[-1]["usage"]["completion_tokens"] == 7


def test_zero_scale_is_exactly_base_forward(srv):
    """The mechanism, not just the bookkeeping: inside _base_weights_request
    the LoRA-wrapped model computes the base model's output bit-for-bit, and
    the adapter's contribution returns afterwards."""
    mx.random.seed(7)
    for _, m in srv.model.named_modules():
        if isinstance(m, LoRALinear):
            m.lora_b = mx.random.normal(m.lora_b.shape) + 0.5  # non-zero delta
    x = mx.random.normal((1, 4))

    def base_forward(v):
        for layer in srv.model.layers:
            v = layer.proj.linear(v)
        return v

    y_adapter = srv.model(x)
    y_base = base_forward(x)
    assert not mx.allclose(y_adapter, y_base).item()
    with srv._base_weights_request() as n:
        assert n == 2
        assert mx.array_equal(srv.model(x), y_base).item()
    assert mx.array_equal(srv.model(x), y_adapter).item()
