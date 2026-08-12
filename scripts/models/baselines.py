import numpy as np
import pandas as pd


def fit_seasonal_naive(y_train: pd.Series) -> pd.Series:
    """Historical mean DART by hour-of-week, learned on train only."""
    how = y_train.index.dayofweek * 24 + y_train.index.hour
    return y_train.groupby(how).mean()


def run_baseline_backtest(
    y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, lag_col: str = "dart_lag_48",
) -> pd.DataFrame:
    te = y_test.notna() & X_test[lag_col].notna()
    X_test, y_test = X_test[te], y_test[te]

    seasonal_map = fit_seasonal_naive(y_train)
    how = pd.Series(X_test.index.dayofweek * 24 + X_test.index.hour, index=X_test.index)

    return pd.DataFrame({
        "actual": y_test,
        "persistence": X_test[lag_col],
        "seasonal_naive": how.map(seasonal_map),
    }, index=X_test.index).sort_index()
