import pandas as pd
import pytest

import scripts.evaluation as evaluation
from scripts.config import GAMMA_NEG_START, GAMMA_POS_START
from scripts.data.data_collection import (
    LOAD_ACTUAL_TEST_PATH,
    LOAD_ACTUAL_TRAIN_PATH,
    LOAD_FORECAST_TEST_PATH,
    LOAD_FORECAST_TRAIN_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from scripts.data.features import build_features
from scripts.data.split import chronological_split
from scripts.evaluation import side_clip
from scripts.models.threshold_search import gamma_grid_search, tau_grid_search


@pytest.fixture(scope="module")
def split():
    train = pd.read_csv(TRAIN_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    test = pd.read_csv(TEST_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    load_forecast = pd.concat([
        pd.read_csv(LOAD_FORECAST_TRAIN_PATH, parse_dates=["interval_start_utc", "publish_time_utc"]),
        pd.read_csv(LOAD_FORECAST_TEST_PATH, parse_dates=["interval_start_utc", "publish_time_utc"]),
    ])
    load_actual = pd.concat([
        pd.read_csv(LOAD_ACTUAL_TRAIN_PATH, index_col="interval_start_utc", parse_dates=True),
        pd.read_csv(LOAD_ACTUAL_TEST_PATH, index_col="interval_start_utc", parse_dates=True),
    ])
    raw = pd.concat([train, test]).sort_index()
    X = build_features(raw, load_forecast, load_actual)
    feature_cols = [c for c in X.columns if c not in ("dart", "dart_feat")]
    valid = X[feature_cols].notna().all(axis=1) & X["dart"].notna()
    return chronological_split(X.loc[valid])


@pytest.fixture
def log(split):
    """Runs a small (fast) version of both Phase 6 grid stages against real
    train/validation data and returns the resulting evaluation log. Small grid
    on purpose — this is checking the logging/no-test-leak mechanism, not
    searching for the actual best (gamma, tau); that's a one-off, not a test.
    """
    evaluation.EVALUATION_LOG.clear()
    tr, va, _te = split
    feature_cols = [c for c in tr.columns if c not in ("dart", "dart_feat")]
    X_train, X_val = tr[feature_cols], va[feature_cols]

    gamma_results = gamma_grid_search(
        X_train, tr["dart"], X_val, va["dart"],
        gamma_pos_values=[GAMMA_POS_START * 0.8, GAMMA_POS_START],
        gamma_neg_values=[GAMMA_NEG_START * 0.8, GAMMA_NEG_START],
    )
    assert len(gamma_results) == 4

    # Fix gamma at the starting point, refit once, then sweep tau.
    from scripts.data.labels import build_labels
    from scripts.models.logistic import fit_spike_classifiers

    y_train = build_labels(tr["dart"], GAMMA_POS_START, GAMMA_NEG_START)
    models = fit_spike_classifiers(X_train, y_train["y_pos"], y_train["y_neg"])
    p_pos_val = pd.Series(models["pos"].predict_proba(X_val)[:, 1], index=X_val.index)
    p_neg_val = pd.Series(models["neg"].predict_proba(X_val)[:, 1], index=X_val.index)

    tau_results = tau_grid_search(
        va["dart"], p_pos_val, p_neg_val,
        tau_pos_values=[0.5, 0.7, 0.9],
        tau_neg_values=[0.5, 0.7, 0.9],
    )
    assert len(tau_results) == 9

    return list(evaluation.EVALUATION_LOG)


def test_test_period_untouched(log):
    # Никакой вызов оценки не должен был затронуть тестовый период до фазы 7
    assert not any(entry["period"] == "test" for entry in log)


def test_log_covers_both_grid_stages(log):
    stages = {entry["stage"] for entry in log}
    assert stages == {"gamma_grid", "tau_grid"}
    assert len(log) == 4 + 9


def test_side_clip_resolves_overlap_by_confidence():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    p_pos = pd.Series([0.9, 0.3, 0.6, 0.2], index=idx)
    p_neg = pd.Series([0.4, 0.8, 0.6, 0.1], index=idx)
    # hour 0: only pos fires; hour 1: only neg fires;
    # hour 2: both fire, tied -> pos wins (>=); hour 3: neither fires.
    inc_mask = pd.Series([True, False, True, False], index=idx)
    dec_mask = pd.Series([False, True, True, False], index=idx)

    inc_clipped, dec_clipped = side_clip(p_pos, p_neg, inc_mask, dec_mask)

    assert list(inc_clipped) == [True, False, True, False]
    assert list(dec_clipped) == [False, True, False, False]
    # never both true for the same hour
    assert not (inc_clipped & dec_clipped).any()


def test_log_evaluation_rejects_unknown_period():
    with pytest.raises(ValueError):
        evaluation.log_evaluation(period="holdout")
