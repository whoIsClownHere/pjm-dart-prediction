import pandas as pd
import pytest

from scripts.data.data_collection import (
    LOAD_ACTUAL_TRAIN_PATH,
    LOAD_FORECAST_TRAIN_PATH,
    TRAIN_DATA_PATH,
)
from scripts.data.features import build_features


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    load_forecast = pd.read_csv(
        LOAD_FORECAST_TRAIN_PATH, parse_dates=["interval_start_utc", "publish_time_utc"]
    )
    load_actual = pd.read_csv(LOAD_ACTUAL_TRAIN_PATH, index_col="interval_start_utc", parse_dates=True)
    return build_features(train, load_forecast, load_actual)


@pytest.fixture(scope="module")
def X(df) -> pd.DataFrame:
    """The feature matrix proper — `dart`/`dart_feat` are schema, not features."""
    return df.drop(columns=["dart", "dart_feat"])


def test_feature_count(X):
    assert X.shape[1] <= 15


def test_no_autoregressive_bloat(X):
    """MIGRATION_PLAN's own version of this test (`c.startswith(banned)` with
    banned including "dart_lag_1", "dart_lag_2", "dart_lag_3") false-positives on
    `dart_lag_24`, since "dart_lag_24" also starts with the string "dart_lag_2" —
    it would fail a column README §4 explicitly keeps. Using exact names for the
    single-digit lags instead, prefix only for rolling stats and rt_lmp lags,
    which is what the check is actually meant to catch.
    """
    banned_exact = {"dart_lag_1", "dart_lag_2", "dart_lag_3"}
    banned_prefix = ("dart_mean", "dart_std", "rt_lmp_lag")
    assert not (set(X.columns) & banned_exact)
    assert not any(c.startswith(banned_prefix) for c in X.columns)
    assert "dart_lag_24" in X.columns
    assert "dart_lag_48" in X.columns


def test_calendar_in_eastern_time(df):
    et = df.index.tz_convert("America/New_York")
    assert (df["hour_of_day"] == et.hour).all()


def test_no_raw_price_levels_in_features(X):
    """rt_lmp/da_lmp are literal components of `dart` (README §4 doesn't list
    them) — Phase 0's audit flagged these as target leakage baked into the old
    feature matrix; must not resurface here."""
    assert "rt_lmp" not in X.columns
    assert "da_lmp" not in X.columns


def test_season_and_peak_are_binary(X):
    for col in ("season_Summer", "season_Winter", "is_peak", "is_weekend", "is_holiday"):
        assert set(X[col].unique()) <= {0, 1, True, False}


def test_no_weekend_or_holiday_hours_are_peak(df):
    et = df.index.tz_convert("America/New_York")
    assert not (df.loc[df["is_weekend"] == 1, "is_peak"] == 1).any()
    assert not (df.loc[df["is_holiday"] == 1, "is_peak"] == 1).any()
