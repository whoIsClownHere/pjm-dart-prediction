import pandas as pd
import pytest

from scripts.config import GAMMA_NEG_START, GAMMA_POS_START
from scripts.data.data_collection import TEST_DATA_PATH, TRAIN_DATA_PATH
from scripts.data.labels import build_labels
from scripts.data.split import chronological_split


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    test = pd.read_csv(TEST_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    return pd.concat([train, test]).sort_index()


@pytest.fixture(scope="module")
def split(raw):
    return chronological_split(raw)


@pytest.fixture(scope="module")
def df(raw) -> pd.DataFrame:
    return raw.join(build_labels(raw["dart"], GAMMA_POS_START, GAMMA_NEG_START))


def test_split_chronological(split):
    tr, va, te = split
    assert tr.index.max() < va.index.min() < va.index.max() < te.index.min()


def test_label_rates(df):
    assert 0.02 < df["y_pos"].mean() < 0.12
    assert 0.005 < df["y_neg"].mean() < 0.06


def test_split_matches_readme_boundaries(split):
    tr, va, te = split
    assert tr.index.min() == pd.Timestamp("2019-01-01", tz="UTC")
    assert tr.index.max() == pd.Timestamp("2023-12-31 23:00", tz="UTC")
    assert va.index.min() == pd.Timestamp("2024-01-01", tz="UTC")
    assert va.index.max() == pd.Timestamp("2024-12-31 23:00", tz="UTC")
    assert te.index.min() == pd.Timestamp("2025-01-01", tz="UTC")
    assert te.index.max() == pd.Timestamp("2026-07-31 23:00", tz="UTC")


def test_split_covers_every_row_exactly_once(raw, split):
    tr, va, te = split
    assert len(tr) + len(va) + len(te) == len(raw)
    assert not (set(tr.index) & set(va.index) & set(te.index))


def test_labels_never_touch_dart(df):
    """Building y_pos/y_neg must not mutate or filter `dart` itself."""
    assert df["dart"].notna().all()
    assert (df.loc[df["y_pos"] == 1, "dart"] >= GAMMA_POS_START).all()
    assert (df.loc[df["y_neg"] == 1, "dart"] <= -GAMMA_NEG_START).all()
