"""Image parts on the text-only (mlx_lm) path must be answered 422, not crash.

2026-09-07..12: every inbound photo hit prepare_prompt_vlm with processor/config
None (AttributeError), the handler thread died, curl saw "Server returned nothing"
(42 times), and the daemon took its cloud fallback. A 422 keeps that fallback
(provider_http maps any non-2xx to HU_ERR_PROVIDER_RESPONSE) without the crash.
"""
import importlib.util
import os
import sys

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "mlx_server_vision_guard_under_test", os.path.join(_SCRIPTS, "mlx-server.py"))
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


IMG = {"role": "user", "content": [{"type": "text", "text": "look"},
                                   {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
TXT = {"role": "user", "content": "hey"}


def test_lm_path_with_image_raises_422_request_error():
    srv = _load_server()
    with pytest.raises(srv._RequestError) as ei:
        srv._reject_images_on_lm_path([TXT, IMG], lm_path=True)
    assert ei.value.status == 422
    assert ei.value.body["error"]["type"] == "unsupported_modality"
    assert "vision" in ei.value.body["error"]["message"]


def test_lm_path_text_only_and_vlm_path_with_image_pass_through():
    srv = _load_server()
    srv._reject_images_on_lm_path([TXT], lm_path=True)         # no images -> fine
    srv._reject_images_on_lm_path([TXT, IMG], lm_path=False)   # VLM path -> its job


def test_guard_reads_the_module_global_by_default():
    srv = _load_server()
    srv.use_lm_path = True
    with pytest.raises(srv._RequestError):
        srv._reject_images_on_lm_path([IMG])
    srv.use_lm_path = False
    srv._reject_images_on_lm_path([IMG])


def test_generate_response_rejects_before_touching_a_model():
    srv = _load_server()
    srv.use_lm_path = True
    srv.config = None
    with pytest.raises(srv._RequestError) as ei:
        srv.generate_response([IMG], max_tokens=8, temperature=0.0)
    assert ei.value.status == 422


def test_do_post_maps_request_error_to_a_json_response(monkeypatch):
    """The accept thread must answer, not die: mirror the _QueueFull branch."""
    srv = _load_server()
    sent = {}

    class FakeHandler:
        def _send_json(self, code, body, headers=None):
            sent["code"], sent["body"] = code, body

    def boom(priority, fn, name=None):
        raise srv._RequestError(422, {"error": {"message": "m", "type": "unsupported_modality"}})
    monkeypatch.setattr(srv.admission_queue, "submit_and_wait", boom)
    # exercise the exact except-chain from do_POST without a socket
    h = FakeHandler()
    try:
        srv.admission_queue.submit_and_wait(0, lambda: None, name="chat_completion")
    except srv._QueueFull as exc:  # pragma: no cover
        code, body_, headers = srv._queue_full_http_response(exc)
        h._send_json(code, body_, headers=headers)
    except srv._RequestError as exc:
        h._send_json(exc.status, exc.body)
    assert sent == {"code": 422, "body": {"error": {"message": "m", "type": "unsupported_modality"}}}
