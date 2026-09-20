"""EmbeddingGemma needs task prefixes; nomic must get none. The daemon marks
requests with input_type (embedder_http.c); the server maps (model, type) -> prefix."""
import importlib.util, os, sys, pytest
pytest.importorskip("mlx.core"); pytest.importorskip("mlx_lm")
_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")

def _load(monkeypatch=None, model=None):
    if model is not None:
        os.environ["HU_EMBED_MODEL_ID"] = model
    else:
        os.environ.pop("HU_EMBED_MODEL_ID", None)
    spec = importlib.util.spec_from_file_location("mlx_server_embed_prefix_under_test", os.path.join(_SCRIPTS, "mlx-server.py"))
    srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv); return srv

def test_default_model_is_nomic_with_no_prefix():
    srv = _load()
    assert "nomic" in srv._EMBED_MODEL_ID
    assert srv._embed_prefix(srv._EMBED_MODEL_ID, "query") == "" and srv._embed_prefix(srv._EMBED_MODEL_ID, "document") == ""

def test_env_selects_gemma_and_prefixes_follow_input_type():
    srv = _load(model="mlx-community/embeddinggemma-300m-8bit")
    assert srv._EMBED_MODEL_ID.endswith("embeddinggemma-300m-8bit")
    assert srv._embed_prefix(srv._EMBED_MODEL_ID, "query") == "task: search result | query: "
    assert srv._embed_prefix(srv._EMBED_MODEL_ID, "document") == "title: none | text: "
    # missing or unknown marker indexes as a document, never as a query
    assert srv._embed_prefix(srv._EMBED_MODEL_ID, None) == "title: none | text: "
    assert srv._embed_prefix(srv._EMBED_MODEL_ID, "weird") == "title: none | text: "
    os.environ.pop("HU_EMBED_MODEL_ID", None)
