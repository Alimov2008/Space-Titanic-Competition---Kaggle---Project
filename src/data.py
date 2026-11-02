import pandas as pd

from config import TARGET_COLUMN, TRAIN_DATA_PATH


def load_data(path=TRAIN_DATA_PATH):
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values")

    return df
