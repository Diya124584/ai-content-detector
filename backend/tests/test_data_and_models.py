import pandas as pd
import pytest

from app.data import check_leakage, split_grouped, split_holdout, split_random, standardize
from app.evaluation import compute_metrics, save_results
from app.feature_model import FeatureTextClassifier


def _df(n=120):
    rows = []
    for i in range(n):
        rows.append({"text": f"human style sentence number {i} about topic {i % 5}", "label": 0, "generator": "human", "domain": "a" if i % 2 else "b", "source": f"s{i}"})
        rows.append({"text": f"ai style passage item {i} regarding subject {i % 5}", "label": 1, "generator": ["g1", "g2", "g3"][i % 3], "domain": "a" if i % 2 else "b", "source": f"s{i}"})
    return standardize(pd.DataFrame(rows))


def test_standardize_fills_columns_and_removes_dupes_and_conflicts():
    raw = pd.DataFrame({"text": ["same", "same", "x y", "conflict", "conflict", ""], "label": [0, 0, 1, 0, 1, 1]})
    out = standardize(raw)
    assert list(out.columns) == ["text", "label", "source", "generator", "domain"]
    assert sorted(out["text"]) == ["same", "x y"]
    assert (out["domain"] == "unknown").all()


def test_standardize_label_map_and_bad_labels():
    out = standardize(pd.DataFrame({"t": ["a b", "c d", "e f"], "y": ["human", "ai", "other"]}), "t", "y", {"human": 0, "ai": 1})
    assert len(out) == 2


def test_random_split_no_leakage():
    tr, va, te = split_random(_df())
    assert sum(check_leakage(tr, va, te).values()) == 0
    assert len(tr) + len(va) + len(te) == len(_df())


def test_unseen_generator_split():
    tr, va, te = split_holdout(_df(), "generator", ["g3"])
    assert set(te["generator"]) == {"g3"}
    assert "g3" not in set(tr["generator"]) | set(va["generator"])


def test_grouped_split_keeps_groups_together():
    tr, va, te = split_grouped(_df(), "source")
    assert not (set(tr["source"]) & set(te["source"])) and not (set(tr["source"]) & set(va["source"]))


def test_holdout_missing_value_raises():
    with pytest.raises(ValueError):
        split_holdout(_df(), "generator", ["nonexistent"])


def test_feature_model_train_save_load_predict(tmp_path):
    df = _df(40)
    m = FeatureTextClassifier.train(df["text"], df["label"], min_df=1)
    m.save(tmp_path / "m.joblib")
    m2 = FeatureTextClassifier.load(tmp_path / "m.joblib")
    p = m2.predict_proba(["ai style passage item 3 regarding subject 3", "human style sentence number 3 about topic 3"])
    assert len(p) == 2 and all(0 <= x <= 1 for x in p)
    assert m2.metadata["label_definition"] == {"0": "human", "1": "ai"}
    assert m2.term_contributions("ai style passage")[0]["term"]


def test_feature_model_rejects_bad_labels():
    with pytest.raises(ValueError):
        FeatureTextClassifier.train(["a", "b"], [0, 0])
    with pytest.raises(ValueError):
        FeatureTextClassifier.train(["a", "b"], [0, 2])


def test_metrics_and_result_files(tmp_path):
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]
    m = compute_metrics(y, p)
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0 and m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert compute_metrics([1, 1], [0.9, 0.8])["roc_auc"] is None
    save_results("unit", m, tmp_path, y, p, plots=False)
    assert (tmp_path / "unit_metrics.json").exists() and (tmp_path / "results_summary.csv").exists()
