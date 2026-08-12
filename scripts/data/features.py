import numpy as np
import pandas as pd

ET = "US/Eastern"
DA_CLOSE_HOUR_ET = 12  # matches scripts/data_collection.py's da_close_hour_et

CUTOFF_ANCHORED_LAG_HOURS = (1, 2, 3, 24)
RT_LMP_CUTOFF_ANCHORED_LAG_HOURS = (1, 24)
TARGET_ANCHORED_LAG_HOURS = (48, 168)
ROLLING_WINDOWS = (24, 168)


def mask_iqr_outliers(
    column: pd.Series,
    iqr_multiplier: float = 1.5,
) -> pd.Series:
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    return column.mask(~column.between(lower_bound, upper_bound))


def last_known_hour_utc(operating_day_et: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Latest hour whose DART/RT LMP is settled and knowable by DA close."""
    cutoff_et = operating_day_et - pd.DateOffset(days=1) + pd.Timedelta(hours=DA_CLOSE_HOUR_ET)
    cutoff_utc = cutoff_et.tz_convert("UTC")
    return cutoff_utc - pd.Timedelta(hours=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the leakage-safe feature matrix from the joined DART/load-forecast columns."""
    df = df.sort_index()
    et = df.index.tz_convert(ET)
    operating_day = et.normalize()

    out = pd.DataFrame(index=df.index)

    hour = et.hour.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = et.dayofweek.to_numpy()
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)

    month = pd.Categorical(et.month, categories=range(1, 13))
    month_dummies = pd.get_dummies(month, prefix="month")
    month_dummies.index = df.index
    out = pd.concat([out, month_dummies], axis=1)

    ref_hour = last_known_hour_utc(operating_day)
    dart = df["dart"]
    rt_lmp = df["rt_lmp"]

    for n in CUTOFF_ANCHORED_LAG_HOURS:
        lookup = ref_hour - pd.Timedelta(hours=n - 1)
        out[f"dart_lag_{n}"] = dart.reindex(lookup).to_numpy()

    for n in RT_LMP_CUTOFF_ANCHORED_LAG_HOURS:
        lookup = ref_hour - pd.Timedelta(hours=n - 1)
        out[f"rt_lmp_lag_{n}"] = rt_lmp.reindex(lookup).to_numpy()

    for n in TARGET_ANCHORED_LAG_HOURS:
        out[f"dart_lag_{n}"] = dart.shift(n).to_numpy()
        out[f"rt_lmp_lag_{n}"] = rt_lmp.shift(n).to_numpy()

    rolling_mean = {w: dart.rolling(w, min_periods=1).mean() for w in ROLLING_WINDOWS}
    rolling_std = {w: dart.rolling(w, min_periods=1).std() for w in ROLLING_WINDOWS}

    out["dart_mean_24"] = rolling_mean[24].reindex(ref_hour).to_numpy()
    out["dart_std_24"] = rolling_std[24].reindex(ref_hour).to_numpy()
    out["dart_mean_168"] = rolling_mean[168].reindex(ref_hour).to_numpy()

    out['rt_lmp'] = df['rt_lmp']
    out['da_lmp'] = df['da_lmp']

    return out.dropna()


