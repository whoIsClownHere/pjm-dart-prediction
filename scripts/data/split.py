import pandas as pd

from scripts.config import TEST_END, TRAIN_END, VALIDATION_END


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """README §6: train 2019-2023, validation 2024, test 2025-2026-07.

    `scripts.config` is the single source of the boundary dates — nothing else
    should hardcode them (MIGRATION_PLAN Phase 4).
    """
    train_end = pd.Timestamp(TRAIN_END, tz="UTC")
    validation_end = pd.Timestamp(VALIDATION_END, tz="UTC")
    test_end = pd.Timestamp(TEST_END, tz="UTC")

    train = df[df.index < train_end]
    validation = df[(df.index >= train_end) & (df.index < validation_end)]
    test = df[(df.index >= validation_end) & (df.index < test_end)]
    return train, validation, test
