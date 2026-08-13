import numpy as np
import pandas as pd
import pytest

from scripts.data.data_collection import (
    LOAD_ACTUAL_TRAIN_PATH,
    LOAD_FORECAST_TRAIN_PATH,
    TRAIN_DATA_PATH,
)
from scripts.data.features import build_load_features, cutoff, select_known_forecast


@pytest.fixture(scope="module")
def delivery_index() -> pd.DatetimeIndex:
    """The real, already-collected train-period DART index (2019-2025)."""
    train = pd.read_csv(TRAIN_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    return train.index


@pytest.fixture(scope="module")
def load_forecast() -> pd.DataFrame:
    return pd.read_csv(
        LOAD_FORECAST_TRAIN_PATH, parse_dates=["interval_start_utc", "publish_time_utc"]
    )


@pytest.fixture(scope="module")
def load_actual() -> pd.DataFrame:
    return pd.read_csv(LOAD_ACTUAL_TRAIN_PATH, index_col="interval_start_utc", parse_dates=True)


@pytest.fixture(scope="module")
def panel(load_forecast, delivery_index) -> pd.DataFrame:
    """Vintage selected for each delivery hour itself (`load_fc_dom`/`rto`)."""
    cutoff_utc = cutoff(delivery_index)
    return select_known_forecast(load_forecast, delivery_index, cutoff_utc)


@pytest.fixture(scope="module")
def df(delivery_index, load_forecast, load_actual) -> pd.DataFrame:
    return build_load_features(delivery_index, load_forecast, load_actual)


def test_no_future_vintage(panel):
    known = panel.dropna(subset=["forecast_publish_utc"])
    assert len(known) / len(panel) > 0.98  # matches test_load_coverage's threshold
    assert (known["forecast_publish_utc"] < known["cutoff_utc"]).all()


def test_load_coverage(df):
    assert df["load_fc_dom"].isna().mean() < 0.02


def test_select_known_forecast_picks_latest_pre_cutoff_vintage():
    """Synthetic, hand-checkable: three vintages for one hour, only the first two
    are before cutoff — the function must pick the later of those two, never the
    one published after cutoff."""
    forecast = pd.DataFrame({
        "interval_start_utc": pd.to_datetime(["2024-01-02T12:00:00Z"] * 3),
        "publish_time_utc": pd.to_datetime([
            "2024-01-01T09:45:00Z",
            "2024-01-01T15:45:00Z",
            "2024-01-01T21:45:00Z",  # at/after cutoff below -> must not be picked
        ]),
        "load_fc_dom": [100.0, 110.0, 999.0],
    })
    target = pd.DatetimeIndex(["2024-01-02T12:00:00Z"])
    as_of = pd.DatetimeIndex(["2024-01-01T21:45:00Z"])  # exactly equal to the 3rd vintage's publish time

    out = select_known_forecast(forecast, target, as_of)

    assert out.loc[0, "load_fc_dom"] == 110.0
    assert out.loc[0, "forecast_publish_utc"] == pd.Timestamp("2024-01-01T15:45:00Z")


def test_select_known_forecast_no_vintage_yet_is_nan():
    forecast = pd.DataFrame({
        "interval_start_utc": pd.to_datetime(["2024-01-02T12:00:00Z"]),
        "publish_time_utc": pd.to_datetime(["2024-01-02T00:00:00Z"]),
        "load_fc_dom": [100.0],
    })
    target = pd.DatetimeIndex(["2024-01-02T12:00:00Z"])
    as_of = pd.DatetimeIndex(["2024-01-01T00:00:00Z"])  # before the only available vintage

    out = select_known_forecast(forecast, target, as_of)

    assert pd.isna(out.loc[0, "load_fc_dom"])
    assert pd.isna(out.loc[0, "forecast_publish_utc"])


def test_load_fc_err_uses_last_known_hour_not_delivery_hour():
    """load_fc_err_dom must score the forecast against the actual for the last
    known hour (cutoff - 1h), not for the delivery hour itself (which hasn't
    happened yet and has no actual)."""
    delivery = pd.DatetimeIndex(["2024-01-02T15:00:00Z"])  # some hour on 2024-01-02
    c = cutoff(delivery)[0]
    ref_hour = c - pd.Timedelta(hours=1)

    forecast = pd.DataFrame({
        "interval_start_utc": [ref_hour, delivery[0]],
        "publish_time_utc": [ref_hour - pd.Timedelta(hours=2)] * 2,
        "load_fc_dom": [50.0, 9999.0],
        "load_fc_rto": [500.0, 99999.0],
    })
    load_actual = pd.DataFrame({
        "load_actual_dom": [45.0],
        "load_actual_rto": [480.0],
    }, index=pd.DatetimeIndex([ref_hour]))

    out = build_load_features(delivery, forecast, load_actual)

    assert out.loc[delivery[0], "load_fc_err_dom"] == pytest.approx(50.0 - 45.0)
    assert out.loc[delivery[0], "load_fc_err_rto"] == pytest.approx(500.0 - 480.0)
