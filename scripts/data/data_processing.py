import pandas as pd

from features import build_features
from data_collection import INPUT_DATA_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH


def save_model_inputs(df: pd.DataFrame, path: str) -> None:
    """Build the feature matrix, drop rows with an extreme DA/RT LMP, pair with the raw `dart` target, and save to CSV."""
    X = build_features(df)

    y = df["dart"].reindex(X.index).rename("dart")
    y.to_frame().join(X).to_csv(path)

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    test = pd.read_csv(TEST_DATA_PATH, index_col="interval_start_utc", parse_dates=True)
    save_model_inputs(train, INPUT_DATA_DIR / "model_inputs.csv")
    save_model_inputs(test, INPUT_DATA_DIR / "model_inputs_test.csv")
