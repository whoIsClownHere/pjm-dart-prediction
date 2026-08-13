import pandas as pd
import pytest

from scripts.data import data_collection
from scripts.data.data_collection import (
    LOAD_ACTUAL_TEST_PATH,
    LOAD_ACTUAL_TRAIN_PATH,
    LOAD_FORECAST_TEST_PATH,
    LOAD_FORECAST_TRAIN_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from scripts.data.features import build_features


@pytest.fixture(scope="module")
def raw_dart() -> pd.Series:
    """Untouched `dart` from the already-collected raw CSVs (train+test, concatenated)."""
    train = pd.read_csv(TRAIN_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    test = pd.read_csv(TEST_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    return pd.concat([train, test]).sort_index()["dart"]


@pytest.fixture(scope="module")
def df(raw_dart) -> pd.DataFrame:
    """Output of the current `build_features`, the dataset acceptance criteria are about."""
    train = pd.read_csv(TRAIN_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    test = pd.read_csv(TEST_DATA_PATH, index_col="interval_start_utc", parse_dates=True)

    load_forecast_train = pd.read_csv(
        LOAD_FORECAST_TRAIN_PATH, parse_dates=["interval_start_utc", "publish_time_utc"]
    )
    load_forecast_test = pd.read_csv(
        LOAD_FORECAST_TEST_PATH, parse_dates=["interval_start_utc", "publish_time_utc"]
    )
    load_actual_train = pd.read_csv(LOAD_ACTUAL_TRAIN_PATH, index_col="interval_start_utc", parse_dates=True)
    load_actual_test = pd.read_csv(LOAD_ACTUAL_TEST_PATH, index_col="interval_start_utc", parse_dates=True)

    return pd.concat([
        build_features(train, load_forecast_train, load_actual_train),
        build_features(test, load_forecast_test, load_actual_test),
    ]).sort_index()


def test_full_hourly_grid(df):
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    assert len(df) == len(expected)


def test_elliott_present(df):
    # Winter Storm Elliott — крупнейшее событие DART в истории PJM
    window = df.loc["2022-12-23":"2022-12-27", "dart"]
    assert window.notna().sum() >= 100


def test_tails_intact(df):
    assert df["dart"].abs().max() > 200      # старый экспорт давал 77
    assert df["dart"].skew() < -0.5          # левый хвост толще правого


def test_target_never_clipped(df, raw_dart):
    assert df["dart"].equals(raw_dart.reindex(df.index))


def test_get_dart_data_full_grid_with_gaps(monkeypatch):
    """Outer join + reindex must materialize hours missing from BOTH da and rt
    as explicit NaN rows, not silently drop them — this is what Phase 1 replaces
    the old dict-of-Series alignment with, and it can't be checked against the
    already-collected CSVs above (they happen to have no gaps), so it's exercised
    here against a monkeypatched client instead of hitting the live API."""
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")

    da = pd.DataFrame({
        "interval_start_utc": idx.delete(2),  # hour 2 missing from DA
        "location": "DOM",
        "lmp": [10.0, 11.0, 13.0, 14.0],
        "energy": [1.0, 1.0, 1.0, 1.0],
        "congestion": [0.0, 0.0, 0.0, 0.0],
        "loss": [0.0, 0.0, 0.0, 0.0],
    })
    rt = pd.DataFrame({
        "interval_start_utc": idx.delete(3),  # hour 3 missing from RT
        "location": "DOM",
        "lmp": [9.0, 10.0, 12.0, 15.0],
        "energy": [1.0, 1.0, 1.0, 1.0],
        "congestion": [0.0, 0.0, 0.0, 0.0],
        "loss": [0.0, 0.0, 0.0, 0.0],
    })

    def fake_get_dataset(dataset, **kwargs):
        return da.copy() if dataset == "pjm_lmp_day_ahead_hourly" else rt.copy()

    monkeypatch.setattr(data_collection.client, "get_dataset", fake_get_dataset)

    out = data_collection.get_dart_data("DOM", "2024-01-01", "2024-01-01T05:00:00")

    assert len(out) == 5
    assert list(out.index) == list(idx)
    assert pd.isna(out.loc[idx[2], "da_lmp"])   # missing from DA only -> NaN, not dropped
    assert pd.isna(out.loc[idx[2], "dart"])     # dart NaN too, not silently computed wrong
    assert pd.isna(out.loc[idx[3], "rt_lmp"])   # missing from RT only -> NaN, not dropped
    assert out["da_lmp"].notna().sum() == 4
    assert out["rt_lmp"].notna().sum() == 4
