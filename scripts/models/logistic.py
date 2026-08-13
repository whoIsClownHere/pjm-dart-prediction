import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_spike_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """One side (`y_pos` or `y_neg`) of README §7's model: standardized features
    into L2-penalized logistic regression, `class_weight="balanced"` instead of
    resampling. Deliberately this simple — README §7 documents that more flexible
    models (RF/GBM/NN, SMOTE) were tried and rejected, not skipped for convenience.
    """
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),  # penalty="l2" is the sklearn default
    )
    model.fit(X_train, y_train)
    return model


def fit_spike_classifiers(
    X_train: pd.DataFrame, y_pos_train: pd.Series, y_neg_train: pd.Series
) -> dict[str, Pipeline]:
    """The two independent logistic regressions README §7 calls for — one per side."""
    return {
        "pos": fit_spike_classifier(X_train, y_pos_train),
        "neg": fit_spike_classifier(X_train, y_neg_train),
    }
