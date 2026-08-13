import pandas as pd


def build_labels(dart: pd.Series, gamma_pos: float, gamma_neg: float) -> pd.DataFrame:
    """README §5: `y_pos` for INC (`dart >= gamma_pos`), `y_neg` for DEC
    (`dart <= -gamma_neg`). Computed on raw, unfiltered `dart` — never `dart_feat`,
    never clipped; the tails are exactly what these labels are meant to catch.
    """
    return pd.DataFrame({
        "y_pos": (dart >= gamma_pos).astype(int),
        "y_neg": (dart <= -gamma_neg).astype(int),
    }, index=dart.index)
