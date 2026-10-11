import pytest

from app.explainability import SentenceOcclusionExplainer
from app.robustness import compare_texts, summarize, synthetic_sentence_shuffle, synthetic_typos
from app.pipeline import TextDetectionPipeline
from app.config import Settings
from .helpers import SAMPLE, make_pipeline


def test_detect_output_format():
    r = make_pipeline().analyze(SAMPLE).to_dict()
    for key in ("prediction", "ai_probability", "human_probability", "detection_score", "signals", "trust", "explanation"):
        assert key in r
    assert abs(r["ai_probability"] + r["human_probability"] - 1) < 1e-9
    assert r["trust"]["status"] == "prototype / not yet calibrated"
    assert r["trust"]["value"] != r["ai_probability"] * 100  # trust is not the probability renamed


def test_pipeline_without_models_still_works_and_warns():
    p = TextDetectionPipeline(Settings())  # no roberta/ppl/feature model at all
    r = p.analyze(SAMPLE)
    assert r.components_active == ["linguistic"]


def test_failing_component_is_reported_not_fatal():
    class Boom:
        is_loaded = False
        def predict(self, t): raise RuntimeError("no internet")
    p = TextDetectionPipeline(Settings(), roberta=Boom())
    r = p.analyze(SAMPLE)
    assert any("roberta unavailable" in w for w in r.warnings)
    assert p.status()["roberta_baseline"] == "failed"


def test_explanations_agree_with_signal_direction():
    r = make_pipeline(p_ai=0.95, ppl=8.0).analyze(SAMPLE)
    rob = next(e for e in r.explanation if e["factor"] == "roberta_baseline")
    assert rob["direction"] == "toward_ai" and "strongly favors AI" in rob["detail"]
    r2 = make_pipeline(p_ai=0.05, ppl=400.0).analyze(SAMPLE)
    rob2 = next(e for e in r2.explanation if e["factor"] == "roberta_baseline")
    assert rob2["direction"] == "toward_human"


def test_short_text_warning():
    r = make_pipeline().analyze("A few words only in this text here.")
    assert any("unreliable" in w for w in r.warnings)


def test_occlusion_explainer():
    items = SentenceOcclusionExplainer(lambda t: min(1.0, len(t) / 500)).explain(SAMPLE)
    assert items and {"sentence", "delta", "direction"} <= set(items[0])
    assert SentenceOcclusionExplainer(lambda t: 0.5).explain("One sentence only.") == []


def test_tfidf_explain_unavailable_raises():
    with pytest.raises(RuntimeError):
        make_pipeline().explain(SAMPLE, "tfidf_terms")
    with pytest.raises(ValueError):
        make_pipeline().explain(SAMPLE, "nope")


def test_robustness_compare_and_summary():
    p_hi, p_lo = make_pipeline(0.9), make_pipeline(0.1)
    calls = iter([p_hi.analyze(SAMPLE), p_lo.analyze(SAMPLE)])
    res = compare_texts(lambda t: next(calls), SAMPLE, SAMPLE + " Edited.", "human_edited_ai")
    assert res.prediction_changed and res.probability_change < 0
    rep = summarize([res], "unit")
    assert rep.n_pairs == 1 and rep.prediction_stability == 0.0


def test_synthetic_perturbations_are_deterministic():
    assert synthetic_typos(SAMPLE, seed=1) == synthetic_typos(SAMPLE, seed=1) != SAMPLE
    assert sorted(synthetic_sentence_shuffle(SAMPLE).split()) == sorted(SAMPLE.split())
