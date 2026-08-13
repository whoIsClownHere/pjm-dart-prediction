"""Values calibrated once and reused everywhere downstream — not hardcoded
inline in the modules that consume them (MIGRATION_PLAN Phase 4)."""

# README §6: chronological three-way split. [start, TRAIN_END) is train,
# [TRAIN_END, VALIDATION_END) is validation, [VALIDATION_END, TEST_END) is test.
# scripts/data/split.py is the only place these should be read from.
TRAIN_END = "2024-01-01"
VALIDATION_END = "2025-01-01"
TEST_END = "2026-08-01"

# README §5: starting point for the (gamma_pos, gamma_neg) grid search in Phase 6,
# not a final calibrated value. gamma_pos ~= Q0.95, gamma_neg ~= |Q0.02| of raw
# (unfiltered) `dart`, computed on the train split only (2019-2023):
#
#   train = raw.loc[raw.index < "2024-01-01", "dart"]
#   gamma_pos_start = round(train.quantile(0.95), 2)   # 21.17
#   gamma_neg_start = round(abs(train.quantile(0.02)), 2)  # 71.19
GAMMA_POS_START = 21.17
GAMMA_NEG_START = 71.19

# Phase 6: chosen from a (gamma_pos, gamma_neg) grid search around the starting
# point above, ranked by total P&L on validation (2024) at tau=0.8, side-clipped.
# The unconstrained search kept improving P&L toward gamma_pos as low as 4-5 —
# but that pushes y_pos's train rate to ~30-40%, which breaks Phase 4's own
# label-rate acceptance bound (0.02 < y_pos.mean() < 0.12): at that point the
# label stops meaning "rare spike" and starts meaning "any hour INC is mildly
# profitable", a different and easier problem that trades on volume, not edge.
# The grid was constrained to the Phase 4 bounds before picking a winner —
# see docs/audit.md-style reasoning in the Phase 6 chat log for the full trace.
GAMMA_POS = 12.0
GAMMA_NEG = 100.0

# Phase 6: (tau_pos, tau_neg) grid search in [0.5, 0.95] (README §8) at the gamma
# above, ranked by total P&L on validation, side-clipped. Chosen by the plan's
# literal rule (argmax validation P&L) — flagged honestly rather than silently:
# this maximum sits at tau_pos=0.5 (INC recall 0.95 on validation), which does
# NOT match README §9's "trades rarely" expectation (recall 0.01-0.15). Neither
# side reaches that recall band anywhere in the searched 0.5-0.95 range for this
# feature set — see the Phase 6 chat log. Kept as-is per explicit instruction to
# follow the plan's selection rule literally rather than override it.
TAU_POS = 0.5
TAU_NEG = 0.75

# README §10: per-trade transaction cost, PJM OATT Schedule 9-3 (Market Support
# Service — explicitly covers INC/DEC/UTC virtual transactions) + Schedule
# 9-PJMSettlement, averaged over the 11 published 2024 monthly rates
# (www.pjm.com, "2024 Schedule 9 Rates" / "2024 Schedule 9-PJMSettlement Rates"):
#
#   MS Component 1   (per MWh)             avg 0.042773
#   MS Component 2   (per bid/offer segment) avg 0.020563
#   Settlement 2d     (per MWh)             avg 0.000724
#   Settlement 2e     (per bid/offer segment) avg 0.000348
#   -------------------------------------------------------
#   TRANSACTION_COST_PER_MWH               = 0.0644  (1 MWh = 1 segment at unit size)
#
# This is a lower bound, not the full picture: it excludes real-time uplift
# allocated to virtuals (README §10's "uplift на виртуалы") — that charge isn't
# a published flat rate, it's a monthly-allocated share of system uplift costs
# and can't be sourced as a single number the way Schedule 9 can. Also excludes
# collateral/margin requirements (README §10's "требования по залогу"), which
# constrain capital, not per-trade P&L, so they don't belong in this deduction.
TRANSACTION_COST_PER_MWH = 0.0644
