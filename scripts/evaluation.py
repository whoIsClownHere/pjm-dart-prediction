import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def make_results(y_true, y_pred, name: str = "model") -> pd.DataFrame:
    """Build a results table from actuals and one model's predictions.

    Both are joined by position, not by index — pass arrays of equal length in
    the same order. Add further models as extra columns.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if name == "actual":
        raise ValueError("'actual' is reserved for the target column")

    index = None
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        index = y_true.index
    elif isinstance(y_pred, (pd.Series, pd.DataFrame)):
        index = y_pred.index

    return pd.DataFrame(
        {"actual": np.asarray(y_true), name: np.asarray(y_pred)},
        index=index,
    )


def score_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE per prediction column against `results["actual"]`.

    MAE_rel normalises MAE by the typical size of |dart|, so the number reads
    as a share rather than as raw $/MWh.
    """
    pred_cols = [c for c in results.columns if c != "actual"]
    scores = {}
    for col in pred_cols:
        valid = results[col].notna() & results["actual"].notna()
        actual = results.loc[valid, "actual"]
        err = results.loc[valid, col] - actual
        mae = err.abs().mean()
        scores[col] = {
            "MAE": mae,
            "RMSE": np.sqrt((err ** 2).mean()),
            "MAE_rel": mae / actual.abs().mean(),
            "n": valid.sum(),
        }
    return pd.DataFrame(scores).T


def score_direction(results: pd.DataFrame) -> pd.DataFrame:
    """Accuracy/precision/recall/F1 per prediction column, treating sign(dart) as a class.

    The positive class is dart > 0. Compare accuracy against the base rate —
    a constant "always positive" prediction already scores that.
    """
    pred_cols = [c for c in results.columns if c != "actual"]
    scores = {}
    for col in pred_cols:
        valid = results[col].notna() & results["actual"].notna()
        actual_class = results.loc[valid, "actual"] > 0
        pred_class = results.loc[valid, col] > 0
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_class, pred_class, average="binary", zero_division=0,
        )
        scores[col] = {
            "accuracy": accuracy_score(actual_class, pred_class),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": valid.sum(),
        }
    return pd.DataFrame(scores).T