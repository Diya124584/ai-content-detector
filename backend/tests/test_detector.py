import os

import pytest

from app.detector import LabelMappingError, RobertaBaseline, resolve_label_mapping


def test_label_mapping_from_names():
    ai, hu, verified, _ = resolve_label_mapping({0: "Fake", 1: "Real"})
    assert (ai, hu, verified) == (0, 1, True)
    ai, hu, verified, _ = resolve_label_mapping({0: "human", 1: "ai"})
    assert (ai, hu) == (1, 0)


def test_generic_labels_are_not_guessed():
    with pytest.raises(LabelMappingError):
        resolve_label_mapping({0: "LABEL_0", 1: "LABEL_1"})


def test_manual_override():
    ai, hu, verified, note = resolve_label_mapping({0: "LABEL_0", 1: "LABEL_1"}, ai_index_override=0)
    assert (ai, hu, verified) == (0, 1, False) and "manually" in note


def test_non_binary_rejected():
    with pytest.raises(LabelMappingError):
        resolve_label_mapping({0: "a", 1: "b", 2: "c"})


def test_baseline_is_lazy_and_labelled_baseline():
    b = RobertaBaseline()
    assert not b.is_loaded and b.role == "pretrained-baseline"
    with pytest.raises(ValueError):
        b.predict("")


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to download/run RoBERTa")
def test_real_roberta_output_format():
    pytest.importorskip("torch")
    out = RobertaBaseline().predict("This is a reasonably ordinary sentence written for a unit test. " * 5)
    assert 0.0 <= out.ai_probability <= 1.0
    assert abs(out.ai_probability + out.human_probability - 1.0) < 1e-6
    assert out.role == "pretrained-baseline"
