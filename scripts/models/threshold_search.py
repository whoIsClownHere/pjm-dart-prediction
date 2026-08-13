import itertools

import pandas as pd

from scripts.data.labels import build_labels
from scripts.evaluation import log_evaluation, score_pnl, side_clip
from scripts.models.logistic import fit_spike_classifiers


def gamma_grid_search(
    X_train: pd.DataFrame,
    dart_train: pd.Series,
    X_val: pd.DataFrame,
    dart_val: pd.Series,
    gamma_pos_values: list[float],
    gamma_neg_values: list[float],
    tau_pos: float = 0.5,
    tau_neg: float = 0.5,
) -> pd.DataFrame:
    """MIGRATION_PLAN Phase 6, stage 1: for each `(gamma_pos, gamma_neg)`, rebuild
    `y_pos`/`y_neg` on train, refit both classifiers, side-clip, and score P&L on
    validation at a fixed `tau` (default 0.5 — just to rank gamma; `tau` itself is
    tuned separately in stage 2 once gamma is fixed). Every point is logged via
    `log_evaluation(period="validation", ...)`.
    """
    rows = []
    for gamma_pos, gamma_neg in itertools.product(gamma_pos_values, gamma_neg_values):
        y_train = build_labels(dart_train, gamma_pos, gamma_neg)
        models = fit_spike_classifiers(X_train, y_train["y_pos"], y_train["y_neg"])

        p_pos = pd.Series(models["pos"].predict_proba(X_val)[:, 1], index=X_val.index)
        p_neg = pd.Series(models["neg"].predict_proba(X_val)[:, 1], index=X_val.index)

        inc_mask, dec_mask = side_clip(p_pos, p_neg, p_pos >= tau_pos, p_neg >= tau_neg)
        pnl = score_pnl(dart_val, inc_mask, dec_mask)

        log_evaluation(
            period="validation", stage="gamma_grid",
            gamma_pos=gamma_pos, gamma_neg=gamma_neg, tau_pos=tau_pos, tau_neg=tau_neg,
            total_pnl=pnl["total_pnl"], sharpe=pnl["sharpe"], n_trades=pnl["n_trades"],
        )
        rows.append({
            "gamma_pos": gamma_pos, "gamma_neg": gamma_neg,
            "total_pnl": pnl["total_pnl"], "sharpe": pnl["sharpe"],
            "n_trades": pnl["n_trades"], "max_drawdown": pnl["max_drawdown"],
        })
    return pd.DataFrame(rows)


def tau_grid_search(
    dart_val: pd.Series,
    p_pos_val: pd.Series,
    p_neg_val: pd.Series,
    tau_pos_values: list[float],
    tau_neg_values: list[float],
) -> pd.DataFrame:
    """MIGRATION_PLAN Phase 6, stage 2: with gamma (and therefore `p_pos_val`/
    `p_neg_val`) already fixed, sweep probability thresholds — no retraining, just
    re-thresholding + side-clipping + P&L on validation. Every point is logged via
    `log_evaluation(period="validation", ...)`.
    """
    rows = []
    for tau_pos, tau_neg in itertools.product(tau_pos_values, tau_neg_values):
        inc_mask, dec_mask = side_clip(p_pos_val, p_neg_val, p_pos_val >= tau_pos, p_neg_val >= tau_neg)
        pnl = score_pnl(dart_val, inc_mask, dec_mask)

        log_evaluation(
            period="validation", stage="tau_grid",
            tau_pos=tau_pos, tau_neg=tau_neg,
            total_pnl=pnl["total_pnl"], sharpe=pnl["sharpe"], n_trades=pnl["n_trades"],
        )
        rows.append({
            "tau_pos": tau_pos, "tau_neg": tau_neg,
            "total_pnl": pnl["total_pnl"], "sharpe": pnl["sharpe"],
            "n_trades": pnl["n_trades"], "max_drawdown": pnl["max_drawdown"],
        })
    return pd.DataFrame(rows)
