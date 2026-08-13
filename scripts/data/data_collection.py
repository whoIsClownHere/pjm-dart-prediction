import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from gridstatusio import GridStatusClient

load_dotenv()

GRIDSTATUS_API_KEY = os.environ["GRIDSTATUS_API_KEY"]
client = GridStatusClient(api_key=GRIDSTATUS_API_KEY)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DATA_DIR = ROOT_DIR / "input_data"

ZONE = "DOM"  # PJM Dominion zone

TRAIN_START = "2019-01-01"
TEST_START = "2026-01-01"
TEST_END = "2026-08-01"
TRAIN_END      = TEST_START
HOLDOUT_START  = TEST_START

TRAIN_DATA_PATH = INPUT_DATA_DIR / "dart_data_train.csv"
TEST_DATA_PATH = INPUT_DATA_DIR / "dart_data_test.csv"

LOAD_ZONES = ("DOM", "RTO")
LOAD_FORECAST_TRAIN_PATH = INPUT_DATA_DIR / "load_forecast_train.csv"
LOAD_FORECAST_TEST_PATH = INPUT_DATA_DIR / "load_forecast_test.csv"
LOAD_ACTUAL_TRAIN_PATH = INPUT_DATA_DIR / "load_actual_train.csv"
LOAD_ACTUAL_TEST_PATH = INPUT_DATA_DIR / "load_actual_test.csv"


def get_dart_data(zone: str, start: str, end: str) -> pd.DataFrame:
    """DART = day-ahead LMP minus real-time LMP, hourly, for one PJM zone.

    Positive DART favors INC offers (sell at DA, buy back cheaper at RT);
    negative DART favors DEC bids.

    DA and RT are joined on the outer union of their timestamps, then reindexed
    onto the full expected hourly grid for [start, end). An hour missing from
    both sources still gets a row (all-NaN) instead of silently disappearing.
    """
    da = client.get_dataset(
        dataset="pjm_lmp_day_ahead_hourly",
        start=start, end=end,
        filter_column="location", filter_value=zone,
        columns=["interval_start_utc", "location", "lmp", "energy", "congestion", "loss"],
    )
    rt = client.get_dataset(
        dataset="pjm_lmp_real_time_hourly",
        start=start, end=end,
        filter_column="location", filter_value=zone,
        columns=["interval_start_utc", "location", "lmp", "energy", "congestion", "loss"],
    )

    da = da.set_index("interval_start_utc")[["lmp", "energy", "congestion", "loss"]].add_prefix("da_")
    rt = rt.set_index("interval_start_utc")[["lmp", "energy", "congestion", "loss"]].add_prefix("rt_")

    full_index = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    out = da.join(rt, how="outer").reindex(full_index)

    out["dart"] = out["da_lmp"] - out["rt_lmp"]
    return out


def save_data(zone: str, start: str, end: str, path: str) -> None:
    """Fetch DART data for one zone and save it to CSV."""
    dart = get_dart_data(zone, start, end)
    dart.to_csv(path)


def collect_train_data() -> None:
    """Fetch/refresh the training window (`TRAIN_START`-`TRAIN_END`, i.e. 2019-2025) into `TRAIN_DATA_PATH`."""
    save_data(ZONE, TRAIN_START, TRAIN_END, TRAIN_DATA_PATH)


def collect_test_data() -> None:
    """Fetch/refresh the holdout (`TEST_START` through `TEST_END`) into `TEST_DATA_PATH`."""
    save_data(ZONE, TEST_START, TEST_END, TEST_DATA_PATH)


def get_load_forecast_data(start: str, end: str) -> pd.DataFrame:
    """Every published load-forecast vintage for the DOM zone and the RTO system
    total, hourly. One row per (interval_start_utc, publish_time_utc) — every
    vintage is kept here; picking the one known as of a given cutoff happens
    downstream in `scripts.data.features`.
    """
    df = client.get_dataset(
        dataset="pjm_load_forecast_hourly_historical",
        start=start, end=end,
        columns=["interval_start_utc", "publish_time_utc", "dom", "rto"],
    )
    return df.rename(columns={"dom": "load_fc_dom", "rto": "load_fc_rto"})


def get_load_actual_data(start: str, end: str) -> pd.DataFrame:
    """Actual hourly metered load (MW) for the DOM zone and the RTO system total.

    No publish/vintage timestamp is exposed by this dataset (unlike the forecast
    above) — `mw` is whatever GridStatus currently has on file for that hour. For
    hours within the last few weeks this can still be `is_verified=False` and
    subject to later revision; that column isn't used to filter here (see
    docs/audit.md) since doing so would blank out the tail of the test period.
    """
    frames = []
    for zone in LOAD_ZONES:
        raw = client.get_dataset(
            dataset="pjm_load_metered_hourly",
            start=start, end=end,
            filter_column="zone", filter_value=zone,
            columns=["interval_start_utc", "mw"],
        )
        frames.append(raw.set_index("interval_start_utc")["mw"].rename(f"load_actual_{zone.lower()}"))
    return pd.concat(frames, axis=1)


def collect_load_forecast_data() -> None:
    """Fetch/refresh published load-forecast vintages for train and test windows."""
    get_load_forecast_data(TRAIN_START, TRAIN_END).to_csv(LOAD_FORECAST_TRAIN_PATH, index=False)
    get_load_forecast_data(TEST_START, TEST_END).to_csv(LOAD_FORECAST_TEST_PATH, index=False)


def collect_load_actual_data() -> None:
    """Fetch/refresh actual metered load for train and test windows."""
    get_load_actual_data(TRAIN_START, TRAIN_END).to_csv(LOAD_ACTUAL_TRAIN_PATH)
    get_load_actual_data(TEST_START, TEST_END).to_csv(LOAD_ACTUAL_TEST_PATH)


if __name__ == "__main__":
    collect_train_data()
    collect_test_data()
    collect_load_forecast_data()
    collect_load_actual_data()
