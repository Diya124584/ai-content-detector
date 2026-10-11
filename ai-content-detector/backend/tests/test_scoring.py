import numpy as np
import pytest

from app.features import extract_features
from app.scoring import (LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN, FusionConfig, RawSignals, classify, fit_fusion_weights,
                         fuse, perplexity_ai_score)
from .helpers import SAMPLE


def test_output_schema():
    r = fuse(RawSignals(0.9, 15.0, extract_features(SAMPLE), 0.7))
    assert r.prediction in {LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN}
    assert abs(r.ai_probability + r.human_probability - 1.0) < 1e-9
    assert set(r.signals) == {"roberta", "perplexity", "burstiness", "ttr", "repetition", "feature_model"}
    assert abs(sum(r.weights_used.values()) - 1.0) < 1e-9


def test_three_state_thresholds():
    cfg = FusionConfig(ai_threshold=0.7, human_threshold=0.3)
    assert classify(0.8, cfg) == LABEL_AI
    assert classify(0.2, cfg) == LABEL_HUMAN
    assert classify(0.5, cfg) == LABEL_UNCERTAIN


def test_weights_are_configurable():
    raw = RawSignals(roberta_ai=1.0, feature_model_ai=0.0)
    only_roberta = fuse(raw, FusionConfig(roberta_weight=1, perplexity_weight=0, linguistic_weight=0, feature_model_weight=0))
    only_feat = fuse(raw, FusionConfig(roberta_weight=0, perplexity_weight=0, linguistic_weight=0, feature_model_weight=1))
    assert only_roberta.detection_score == 1.0 and only_feat.detection_score == 0.0


def test_missing_signals_renormalise_and_no_signals_is_uncertain():
    r = fuse(RawSignals(roberta_ai=0.9))
    assert r.weights_used == {"roberta": 1.0}
    empty = fuse(RawSignals())
    assert empty.prediction == LABEL_UNCERTAIN and empty.ai_probability == 0.5


def test_short_text_gates_linguistic_features():
    r = fuse(RawSignals(features=extract_features("Too short.")))
    assert r.signals["ttr"] is None and r.signals["burstiness"] is None


def test_perplexity_direction():
    cfg = FusionConfig()
    assert perplexity_ai_score(5.0, cfg) > perplexity_ai_score(200.0, cfg)
    assert perplexity_ai_score(None, cfg) is None and perplexity_ai_score(-1, cfg) is None


def test_calibrator_changes_probability_not_detection_score():
    r = fuse(RawSignals(roberta_ai=0.9), calibrator=lambda p: 0.6)
    assert r.detection_score == pytest.approx(0.9) and r.ai_probability == pytest.approx(0.6)


def test_invalid_config_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"ai_threshold": 0.2, "human_threshold": 0.8}')
    with pytest.raises(ValueError):
        FusionConfig.load(p)


def test_config_roundtrip(tmp_path):
    cfg = FusionConfig(roberta_weight=0.9)
    cfg.save(tmp_path / "c.json")
    assert FusionConfig.load(tmp_path / "c.json").roberta_weight == 0.9


def test_fit_fusion_weights_prefers_informative_signal():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 300)
    X = np.column_stack([y * 0.6 + rng.rand(300) * 0.4, rng.rand(300), rng.rand(300), rng.rand(300)])
    cfg = fit_fusion_weights(X, y)
    w = cfg.weights()
    assert w["roberta"] == max(w.values()) and sum(w.values()) == pytest.approx(1.0)
