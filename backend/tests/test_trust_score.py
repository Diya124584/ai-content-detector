import numpy as np
import pytest

from app.features import extract_features
from app.trust_score import (PROTOTYPE_STATUS, CalibrationReport, LogitCalibrator, RobustnessReport, brier_score,
                             build_calibration_report, compute_trust, expected_calibration_error, fit_temperature)
from .helpers import SAMPLE


def test_trust_is_separate_from_probability():
    f = extract_features(SAMPLE)
    confident = compute_trust(0.95, {"a": 0.95, "b": 0.9}, f)
    unsure = compute_trust(0.5, {"a": 0.95, "b": 0.05}, f)
    assert 0 <= confident.value <= 100 and confident.value > unsure.value
    short = compute_trust(0.95, {"a": 0.95, "b": 0.9}, extract_features("Tiny text here now ok"))
    assert short.value < confident.value  # same probability, lower trust
    assert any("short_text" in x for x in short.ood_flags)


def test_prototype_status_without_artifacts():
    t = compute_trust(0.7, {"a": 0.7}, extract_features(SAMPLE))
    assert t.status == PROTOTYPE_STATUS
    assert {"calibration", "robustness"} <= set(t.missing_components)


def test_status_changes_with_artifacts():
    f = extract_features(SAMPLE)
    cal = CalibrationReport(100, 0.05, 0.1, 1.0, "toy")
    rob = RobustnessReport(10, 0.9, 0.05, "toy")
    assert compute_trust(0.7, {"a": 0.7}, f, calibration=cal).status != PROTOTYPE_STATUS
    both = compute_trust(0.7, {"a": 0.7}, f, calibration=cal, robustness=rob)
    assert "calibration" not in both.missing_components and "robustness" not in both.missing_components
    assert both.status != compute_trust(0.7, {"a": 0.7}, f, calibration=cal).status


def test_calibration_metrics():
    assert brier_score([1, 0], [1, 0]) == 0.0
    assert expected_calibration_error([0.9] * 10, [1] * 10) == pytest.approx(0.1)
    assert expected_calibration_error([1.0, 0.0], [1, 0]) == 0.0


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.RandomState(0)
    true_p = rng.rand(2000)
    y = (rng.rand(2000) < true_p).astype(int)
    overconfident = 1 / (1 + np.exp(-4 * (np.log(true_p / (1 - true_p + 1e-9) + 1e-9))))
    t = fit_temperature(overconfident, y)
    assert t > 1.5
    rep, cal = build_calibration_report(overconfident, y, "synthetic")
    assert rep.ece < expected_calibration_error(overconfident, y)


def test_identity_calibrator_and_roundtrip(tmp_path):
    c = LogitCalibrator()
    assert c(0.7) == pytest.approx(0.7, abs=1e-6)
    LogitCalibrator(2.0, 0.1).save(tmp_path / "c.json")
    assert LogitCalibrator.load(tmp_path / "c.json").temperature == 2.0
