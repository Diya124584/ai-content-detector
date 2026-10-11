import os

import pytest

from app.perplexity import PerplexityScorer


def test_empty_text_returns_none_without_loading_model():
    s = PerplexityScorer("gpt2")
    r = s.compute("   ")
    assert r.perplexity is None and r.note == "empty text"
    assert not s.is_loaded  # model must not be loaded for empty input


def test_model_is_lazy():
    assert PerplexityScorer("gpt2").is_loaded is False


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to download/run GPT-2")
def test_real_gpt2_perplexity_is_positive_and_model_loaded_once():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    s = PerplexityScorer("gpt2")
    r = s.compute("The quick brown fox jumps over the lazy dog near the river bank today.")
    assert r.perplexity is not None and r.perplexity > 1.0
    model_obj = s._model
    s.compute("Another sentence that is long enough to be scored by the language model.")
    assert s._model is model_obj
    short = s.compute("Hi")
    assert short.perplexity is None
