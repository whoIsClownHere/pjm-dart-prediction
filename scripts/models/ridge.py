import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_ridge_backtest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alpha: float | None = None,
) -> pd.DataFrame:
    """Fit once on train, predict once on test. No refit inside the holdout."""
    assert list(X_train.columns) == list(X_test.columns)
    assert X_train.index.max() < X_test.index.min()

    tr = X_train.notna().all(axis=1) & y_train.notna()
    te = X_test.notna().all(axis=1) & y_test.notna()

    est = Ridge(alpha=alpha) if alpha is not None else RidgeCV(alphas=np.logspace(-2, 4, 25))
    model = make_pipeline(StandardScaler(), est)
    model.fit(X_train[tr], y_train[tr])

    return pd.DataFrame({
        "actual": y_test[te],
        "ridge": model.predict(X_test[te]),
    }, index=X_test.index[te]).sort_index()