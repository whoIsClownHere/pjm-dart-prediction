import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import precision_recall_fscore_support

# MIGRATION_PLAN Phase 6 acceptance: every evaluation call is logged with which
# split it touched, so `not any(entry["period"] == "test" for entry in log)` can
# be checked mechanically rather than by promise. Cleared per grid-search run by
# the caller (see `scripts.models.threshold_search`) — this module doesn't own
# when it's reset, only the recording.
EVALUATION_LOG: list[dict] = []


def log_evaluation(period: str, **fields) -> dict:
    """Record one evaluation call. `period` must be "train", "validation", or
    "test" — Phase 6 must never log "test" until Phase 7's single run."""
    if period not in ("train", "validation", "test"):
        raise ValueError(f"unknown period {period!r}")
    entry = {"period": period, **fields}
    EVALUATION_LOG.append(entry)
    return entry


def side_clip(
    p_pos: pd.Series, p_neg: pd.Series, inc_mask: pd.Series, dec_mask: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """README §6/Phase 6: when both `y_pos` and `y_neg` fire for the same hour,
    keep only the side "согласованное со знаком предсказанного DART". This setup
    never produces a point forecast of `dart` itself (it's classification, not
    regression), so the proxy used here is the classifier's own confidence: keep
    whichever side has the higher probability, drop the other. Hours where only
    one side fired are untouched.
    """
    both = inc_mask & dec_mask
    inc_clipped = inc_mask & (~both | (p_pos >= p_neg))
    dec_clipped = dec_mask & (~both | (p_neg > p_pos))
    return inc_clipped, dec_clipped


def score_spikes(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Precision/recall/F1 for one side (INC or DEC) of the spike classifier.

    README §9: recall is expected to be low (0.01-0.15) by design — the model
    trades rarely and precisely rather than on volume. Don't chase recall here;
    raising it costs precision, and precision is what P&L depends on.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_trades": int(np.asarray(y_pred).sum()),
        "base_rate": float(np.asarray(y_true).mean()),
    }


def trade_masks(p_pos: pd.Series, tau_pos: float, p_neg: pd.Series, tau_neg: float) -> tuple[pd.Series, pd.Series]:
    """README §8's raw trading rule, no side-clipping (that's Phase 6): trade INC
    where `p_pos >= tau_pos`, DEC where `p_neg >= tau_neg`. The two masks can
    overlap — a caller that needs side-clipping resolves that itself."""
    return p_pos >= tau_pos, p_neg >= tau_neg


def score_pnl(
    dart: pd.Series,
    inc_mask: pd.Series,
    dec_mask: pd.Series,
    cost_per_mwh: float = 0.0,
) -> dict:
    """Cumulative P&L, annualized Sharpe, max drawdown, and trade count for a
    given set of INC/DEC trade decisions (README §8: unit size — 1 MWh per trade,
    `pnl = +dart` for INC, `-dart` for DEC). `inc_mask`/`dec_mask` are the actual
    trade signals; if side-clipping applies, the caller (Phase 6) must resolve
    overlapping INC/DEC signals before calling this.

    `cost_per_mwh` (README §10) is deducted from each individual trade's P&L
    before anything is aggregated — "вычитать из каждого трейда, а не из итога" —
    not subtracted as one lump sum from the total at the end.

    Sharpe is computed on daily-aggregated P&L (`mean / std * sqrt(365)`) since
    trades are sparse and hourly — a daily basis is the smallest window where
    "no trade that hour" doesn't just mean a zero return dominating the ratio.
    """
    pnl = pd.Series(0.0, index=dart.index)
    pnl.loc[inc_mask] += dart.loc[inc_mask] - cost_per_mwh
    pnl.loc[dec_mask] += -dart.loc[dec_mask] - cost_per_mwh

    cumulative = pnl.cumsum()
    daily = pnl.resample("D").sum()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else float("nan")
    max_drawdown = (cumulative.cummax() - cumulative).max()

    return {
        "total_pnl": pnl.sum(),
        "cumulative_pnl": cumulative,
        "pnl_series": pnl,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_trades": int(inc_mask.sum() + dec_mask.sum()),
        "n_inc": int(inc_mask.sum()),
        "n_dec": int(dec_mask.sum()),
    }


def pnl_by_year(dart: pd.Series, inc_mask: pd.Series, dec_mask: pd.Series, cost_per_mwh: float = 0.0) -> pd.DataFrame:
    """README §9.3: cumulative P&L by year, INC and DEC counted separately."""
    pnl = pd.Series(0.0, index=dart.index)
    pnl.loc[inc_mask] = dart.loc[inc_mask] - cost_per_mwh
    pnl.loc[dec_mask] = -dart.loc[dec_mask] - cost_per_mwh

    inc_pnl = pnl.where(inc_mask, 0.0)
    dec_pnl = pnl.where(dec_mask, 0.0)
    years = dart.index.year

    return pd.DataFrame({
        "inc_pnl": inc_pnl.groupby(years).sum(),
        "dec_pnl": dec_pnl.groupby(years).sum(),
        "n_inc": inc_mask.groupby(years).sum(),
        "n_dec": dec_mask.groupby(years).sum(),
    })


def baseline_always_trade(dart: pd.Series, side: str, cost_per_mwh: float = 0.0) -> dict:
    """README §9 baseline: always INC or always DEC, unit size, every hour."""
    if side not in ("inc", "dec"):
        raise ValueError(side)
    all_hours = pd.Series(True, index=dart.index)
    none = pd.Series(False, index=dart.index)
    if side == "inc":
        return score_pnl(dart, all_hours, none, cost_per_mwh=cost_per_mwh)
    return score_pnl(dart, none, all_hours, cost_per_mwh=cost_per_mwh)


def baseline_random_trade(
    dart: pd.Series,
    n_inc: int,
    n_dec: int,
    cost_per_mwh: float = 0.0,
    n_simulations: int = 1000,
    seed: int = 0,
) -> dict:
    """README §9 baseline: random trading at the model's own frequency — `n_inc`
    INC trades and `n_dec` DEC trades placed on uniformly random, non-overlapping
    hours. Averaged over `n_simulations` draws (one random draw would just be
    noise); returns the mean/std of `total_pnl` across simulations.
    """
    rng = np.random.default_rng(seed)
    n = len(dart)
    totals = np.empty(n_simulations)
    for i in range(n_simulations):
        chosen = rng.choice(n, size=min(n_inc + n_dec, n), replace=False)
        inc_idx, dec_idx = chosen[:n_inc], chosen[n_inc:n_inc + n_dec]
        inc_mask = pd.Series(False, index=dart.index)
        dec_mask = pd.Series(False, index=dart.index)
        inc_mask.iloc[inc_idx] = True
        dec_mask.iloc[dec_idx] = True
        totals[i] = score_pnl(dart, inc_mask, dec_mask, cost_per_mwh=cost_per_mwh)["total_pnl"]
    return {
        "mean_pnl": float(totals.mean()),
        "std_pnl": float(totals.std()),
        "n_simulations": n_simulations,
        "n_inc": n_inc,
        "n_dec": n_dec,
    }


def js_divergence_calendar(selected: pd.DatetimeIndex, reference: pd.DatetimeIndex, by: str) -> float:
    """README §9.5: Jensen-Shannon divergence between the `by` ("hour" of day,
    or "month" of year, both in ET) distribution of `selected` hours and of
    `reference` hours. Lower means more similar. NaN if either set is empty.
    """
    if by == "hour":
        bins = list(range(24))
        extract = lambda idx: idx.tz_convert("America/New_York").hour  # noqa: E731
    elif by == "month":
        bins = list(range(1, 13))
        extract = lambda idx: idx.tz_convert("America/New_York").month  # noqa: E731
    else:
        raise ValueError(f"by must be 'hour' or 'month', got {by!r}")

    if len(selected) == 0 or len(reference) == 0:
        return float("nan")

    p = pd.Series(extract(selected)).value_counts(normalize=True).reindex(bins, fill_value=0.0)
    q = pd.Series(extract(reference)).value_counts(normalize=True).reindex(bins, fill_value=0.0)
    return float(jensenshannon(p.to_numpy(), q.to_numpy(), base=2))


def spike_timing_divergence(
    p_side: pd.Series, side_mask: pd.Series, reference_mask: pd.Series, by: str
) -> dict:
    """README §9.5: JS divergence between the model's selected-hour calendar
    distribution and the actual-spike calendar distribution — computed both
    against every hour the model traded (`js_all`) and against only its most
    confident top 20% by `p_side` (`js_top20`). `js_top20 < js_all` supports
    "the model's confidence tracks genuine spike timing", not just guessing
    peak calendar slots — README's own example: JS(top 20%)=0.039 vs
    JS(all)=0.094 for DEC.
    """
    selected_all = side_mask[side_mask].index
    reference = reference_mask[reference_mask].index

    if side_mask.any():
        cutoff = p_side[side_mask].quantile(0.8)
        selected_top20 = p_side[side_mask & (p_side >= cutoff)].index
    else:
        selected_top20 = pd.DatetimeIndex([])

    return {
        "js_all": js_divergence_calendar(selected_all, reference, by=by),
        "js_top20": js_divergence_calendar(selected_top20, reference, by=by),
    }
