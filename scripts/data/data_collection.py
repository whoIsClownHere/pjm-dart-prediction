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


def get_dart_data(zone: str, start: str, end: str) -> pd.DataFrame:
    """DART = day-ahead LMP minus real-time LMP, hourly, for one PJM zone.

    Positive DART favors INC offers (sell at DA, buy back cheaper at RT);
    negative DART favors DEC bids.
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

    da = da.set_index("interval_start_utc")
    rt = rt.set_index("interval_start_utc")

    out = pd.DataFrame({
        "da_lmp": da["lmp"],
        "rt_lmp": rt["lmp"],
        "da_energy": da["energy"],
        "rt_energy": rt["energy"],
        "da_congestion": da["congestion"],
        "rt_congestion": rt["congestion"],
        "da_loss": da["loss"],
        "rt_loss": rt["loss"],
    })
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


if __name__ == "__main__":
    collect_train_data()
    collect_test_data()
