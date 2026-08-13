import numpy as np
import pandas as pd
from dateutil.relativedelta import MO, TH
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import DateOffset

from scripts.data.data_collection import INPUT_DATA_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH

ET = "US/Eastern"

TARGET_LAG_HOURS = (24, 48)

WINTER_MONTHS = {12, 1, 2}
SUMMER_MONTHS = {6, 7, 8}
SEASON_CATEGORIES = ["Shoulder", "Summer", "Winter"]

# PJM 5x16 On-Peak: Mon-Fri, non-NERC-holiday, ET hour in [7, 23).
PEAK_START_HOUR_ET = 7
PEAK_END_HOUR_ET = 23


class NERCHolidayCalendar(AbstractHolidayCalendar):
    """The 6 NERC holidays used across PJM/ISO tariffs (README §2). Unlike US
    federal holidays, these are observed on their actual calendar date — never
    shifted for falling on a weekend."""

    rules = [
        Holiday("New Year's Day", month=1, day=1),
        Holiday("Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday("Independence Day", month=7, day=4),
        Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving Day", month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday("Christmas Day", month=12, day=25),
    ]


def winsorize_causal(
    series: pd.Series,
    window: str = "30D",
    iqr_multiplier: float = 1.5,
) -> pd.Series:
    """Clip `series` to trailing Tukey IQR bounds computed strictly from the past.

    Bounds come from `shift(1).rolling(window)` quantiles, so the bound for a given
    hour never uses that hour or any later one. Clips values, never drops rows.
    Before there's a full `window` of history, bounds are NaN and values pass
    through unclipped rather than being dropped.
    """
    past = series.shift(1)
    q1 = past.rolling(window).quantile(0.25)
    q3 = past.rolling(window).quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    return series.clip(lower=lower_bound, upper=upper_bound)


def cutoff(delivery_ts_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """README §1: DA bidding for `delivery_ts_utc` closes 11:00 EPT the day before.

    Single source of truth for "known before the market closed" — used both by
    the load-forecast merge and by every calendar/lag feature in `build_features`.
    """
    et = delivery_ts_utc.tz_convert("America/New_York")
    return (et.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=11)).tz_convert("UTC")


def select_known_forecast(
    forecast: pd.DataFrame,
    target_hours: pd.DatetimeIndex,
    as_of: pd.DatetimeIndex,
) -> pd.DataFrame:
    """For each row i, the row of `forecast` for `interval_start_utc ==
    target_hours[i]` with the latest `publish_time_utc` strictly before
    `as_of[i]`. Never returns a vintage published at or after `as_of[i]`; NaN
    where no such vintage exists yet.

    Returns one row per input, in input order, with `forecast_publish_utc` and
    `cutoff_utc` columns plus whatever value columns `forecast` had (besides
    `interval_start_utc`/`publish_time_utc`).
    """
    lookup = pd.DataFrame({
        "_row": np.arange(len(target_hours)),
        "interval_start_utc": pd.DatetimeIndex(target_hours),
        "cutoff_utc": pd.DatetimeIndex(as_of),
    })
    merged = lookup.merge(forecast, on="interval_start_utc", how="left")
    known = merged[merged["publish_time_utc"] < merged["cutoff_utc"]]
    latest = known.sort_values("publish_time_utc").groupby("_row", as_index=False).tail(1)

    out = lookup[["_row"]].merge(latest, on="_row", how="left").sort_values("_row")
    out = out.drop(columns="_row").rename(columns={"publish_time_utc": "forecast_publish_utc"})
    return out.reset_index(drop=True)


def build_load_features(
    delivery_index: pd.DatetimeIndex,
    load_forecast: pd.DataFrame,
    load_actual: pd.DataFrame,
) -> pd.DataFrame:
    """`load_fc_dom`/`load_fc_rto` (forecast for the delivery hour itself) and
    `load_fc_err_dom`/`load_fc_err_rto` (forecast error for the last hour known
    before cutoff), all vintage-selected via `select_known_forecast` so no row
    ever uses a forecast published at or after that row's own cutoff.
    """
    cutoff_utc = cutoff(delivery_index)

    own_hour = select_known_forecast(load_forecast, delivery_index, cutoff_utc)

    ref_hour = cutoff_utc - pd.Timedelta(hours=1)
    at_ref = select_known_forecast(load_forecast, ref_hour, cutoff_utc)
    actual_at_ref_dom = load_actual["load_actual_dom"].reindex(ref_hour).to_numpy()
    actual_at_ref_rto = load_actual["load_actual_rto"].reindex(ref_hour).to_numpy()

    out = pd.DataFrame(index=delivery_index)
    out["load_fc_dom"] = own_hour["load_fc_dom"].to_numpy()
    out["load_fc_rto"] = own_hour["load_fc_rto"].to_numpy()
    out["load_fc_err_dom"] = at_ref["load_fc_dom"].to_numpy() - actual_at_ref_dom
    out["load_fc_err_rto"] = at_ref["load_fc_rto"].to_numpy() - actual_at_ref_rto
    return out


def _season(month: np.ndarray) -> pd.Categorical:
    labels = np.where(
        np.isin(month, list(WINTER_MONTHS)), "Winter",
        np.where(np.isin(month, list(SUMMER_MONTHS)), "Summer", "Shoulder"),
    )
    return pd.Categorical(labels, categories=SEASON_CATEGORIES)


def build_features(
    df: pd.DataFrame,
    load_forecast: pd.DataFrame,
    load_actual: pd.DataFrame,
) -> pd.DataFrame:
    """README §4's feature set, nothing else: `dart_lag_24`/`dart_lag_48`
    (row-anchored, from `dart_feat`), the four load-forecast columns, and
    calendar (`hour_of_day`, `month`, `is_weekend`, `is_holiday`, season,
    peak flag). `dart`/`dart_feat` are carried through untouched — schema, not
    features.

    Returns every row of `df`, including rows with NaN from lag warmup or from
    an hour genuinely missing in `df`. Dropping those is `save_model_inputs`'s
    job, done once at the end and logged separately for features vs. target.
    """
    df = df.sort_index()
    et = df.index.tz_convert(ET)

    out = pd.DataFrame(index=df.index)

    dart_feat = winsorize_causal(df["dart"])
    for n in TARGET_LAG_HOURS:
        out[f"dart_lag_{n}"] = dart_feat.shift(n).to_numpy()

    load_feats = build_load_features(df.index, load_forecast, load_actual)
    out = pd.concat([out, load_feats], axis=1)

    hour = et.hour.to_numpy()
    dow = et.dayofweek.to_numpy()
    month = et.month.to_numpy()

    holidays = NERCHolidayCalendar().holidays(start=et.min().tz_localize(None), end=et.max().tz_localize(None))
    is_holiday = et.normalize().tz_localize(None).isin(holidays)

    out["hour_of_day"] = hour
    out["month"] = month
    out["is_weekend"] = (dow >= 5).astype(int)
    out["is_holiday"] = is_holiday.astype(int)

    season_dummies = pd.get_dummies(_season(month), prefix="season", drop_first=True)
    season_dummies.index = df.index
    out = pd.concat([out, season_dummies], axis=1)

    is_peak = (dow < 5) & ~is_holiday & (hour >= PEAK_START_HOUR_ET) & (hour < PEAK_END_HOUR_ET)
    out["is_peak"] = is_peak.astype(int)

    out["dart_feat"] = dart_feat
    out["dart"] = df["dart"]

    return out


def save_model_inputs(
    df: pd.DataFrame,
    load_forecast: pd.DataFrame,
    load_actual: pd.DataFrame,
    path: str,
) -> None:
    """Build the feature matrix and save it, dropping incomplete rows at the end.

    Feature-column and target completeness are dropped and counted separately: a
    row missing a feature (lag warmup, or a not-yet-known load forecast) is a
    different failure mode from a row missing `dart` (a genuinely uncollected
    hour), and the two should never be conflated into one opaque `dropna()`.
    """
    X = build_features(df, load_forecast, load_actual)

    feature_cols = [c for c in X.columns if c not in ("dart", "dart_feat")]
    n_total = len(X)
    feature_valid = X[feature_cols].notna().all(axis=1)
    target_valid = X["dart"].notna()

    print(f"{path}")
    print(f"  rows before dropna:        {n_total}")
    print(f"  dropped for feature NaN:   {n_total - int(feature_valid.sum())}")
    print(f"  dropped for missing dart:  {n_total - int(target_valid.sum())}")

    X = X.loc[feature_valid & target_valid]
    print(f"  rows kept:                 {len(X)}")

    X.to_csv(path)

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    test = pd.read_csv(TEST_DATA_PATH, index_col="interval_start_utc", parse_dates=True)

    load_forecast_train = pd.read_csv(
        INPUT_DATA_DIR / "load_forecast_train.csv", parse_dates=["interval_start_utc", "publish_time_utc"]
    )
    load_forecast_test = pd.read_csv(
        INPUT_DATA_DIR / "load_forecast_test.csv", parse_dates=["interval_start_utc", "publish_time_utc"]
    )
    load_actual_train = pd.read_csv(
        INPUT_DATA_DIR / "load_actual_train.csv", index_col="interval_start_utc", parse_dates=True
    )
    load_actual_test = pd.read_csv(
        INPUT_DATA_DIR / "load_actual_test.csv", index_col="interval_start_utc", parse_dates=True
    )

    save_model_inputs(train, load_forecast_train, load_actual_train, INPUT_DATA_DIR / "train_features.csv")
    save_model_inputs(test, load_forecast_test, load_actual_test, INPUT_DATA_DIR / "test_features.csv")
