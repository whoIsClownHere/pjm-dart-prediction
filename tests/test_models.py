import numpy as np
import pandas as pd
import pytest

from scripts.config import GAMMA_NEG_START, GAMMA_POS_START
from scripts.data.data_collection import TEST_DATA_PATH, TRAIN_DATA_PATH
from scripts.data.data_collection import (
    LOAD_ACTUAL_TEST_PATH,
    LOAD_ACTUAL_TRAIN_PATH,
    LOAD_FORECAST_TEST_PATH,
    LOAD_FORECAST_TRAIN_PATH,
)
from scripts.data.features import build_features
from scripts.data.labels import build_labels
from scripts.data.split import chronological_split
from scripts.evaluation import score_pnl, score_spikes, trade_masks
from scripts.models.logistic import fit_spike_classifiers


@pytest.fixture(scope="module")
def feature() -> pd.DataFrame:
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
    return X.loc[valid]


@pytest.fixture(scope="module")
def split(feature):
    return chronological_split(feature)


@pytest.fixture(scope="module")
def models_and_validation(split):
    """Train once on train only, predict on validation only — module-scoped so
    every test below reuses the same fit instead of retraining."""
    tr, va, _te = split
    feature_cols = [c for c in tr.columns if c not in ("dart", "dart_feat")]
    X_train, X_val = tr[feature_cols], va[feature_cols]

    y_train = build_labels(tr["dart"], GAMMA_POS_START, GAMMA_NEG_START)
    y_val = build_labels(va["dart"], GAMMA_POS_START, GAMMA_NEG_START)

    models = fit_spike_classifiers(X_train, y_train["y_pos"], y_train["y_neg"])

    p_pos_val = pd.Series(models["pos"].predict_proba(X_val)[:, 1], index=X_val.index)
    p_neg_val = pd.Series(models["neg"].predict_proba(X_val)[:, 1], index=X_val.index)

    return {
        "models": models,
        "va": va,
        "y_val": y_val,
        "p_pos_val": p_pos_val,
        "p_neg_val": p_neg_val,
    }


def test_models_train_only_on_train_period(split):
    """Sanity check on the fixture itself: train/validation don't overlap and
    train never reaches into validation's date range."""
    tr, va, _te = split
    assert tr.index.max() < va.index.min()


def test_inc_precision_beats_base_rate(models_and_validation):
    """MIGRATION_PLAN Phase 5 acceptance: on validation, precision for the INC
    side must be distinguishable from the base class rate — i.e. the model is
    finding real signal, not just reproducing the class imbalance."""
    d = models_and_validation
    pred_pos = (d["p_pos_val"] >= 0.5).astype(int)
    scores = score_spikes(d["y_val"]["y_pos"], pred_pos)
    assert scores["precision"] > scores["base_rate"]


def test_dec_precision_beats_base_rate(models_and_validation):
    d = models_and_validation
    pred_neg = (d["p_neg_val"] >= 0.5).astype(int)
    scores = score_spikes(d["y_val"]["y_neg"], pred_neg)
    assert scores["precision"] > scores["base_rate"]


def test_predict_proba_in_unit_range(models_and_validation):
    d = models_and_validation
    assert d["p_pos_val"].between(0, 1).all()
    assert d["p_neg_val"].between(0, 1).all()


def test_score_pnl_matches_hand_computed_unit_size():
    dart = pd.Series(
        [10.0, -20.0, 5.0, -3.0],
        index=pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
    )
    inc_mask = pd.Series([True, False, False, False], index=dart.index)
    dec_mask = pd.Series([False, True, False, False], index=dart.index)

    result = score_pnl(dart, inc_mask, dec_mask)

    # INC on hour 0: +10; DEC on hour 1: -(-20) = +20; no trade on hours 2-3.
    assert result["total_pnl"] == pytest.approx(30.0)
    assert result["n_trades"] == 2
    assert result["n_inc"] == 1
    assert result["n_dec"] == 1


def test_trade_masks_thresholding():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    p_pos = pd.Series([0.9, 0.3, 0.71, 0.5], index=idx)
    p_neg = pd.Series([0.1, 0.8, 0.2, 0.6], index=idx)

    inc, dec = trade_masks(p_pos, 0.7, p_neg, 0.7)

    assert list(inc) == [True, False, True, False]
    assert list(dec) == [False, True, False, False]
