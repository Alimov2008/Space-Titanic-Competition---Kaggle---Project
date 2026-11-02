from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numeric_features = [c for c in X.columns if c not in categorical_features]

    numeric_pipeline = Pipeline(
        [
            ("passthrough", "passthrough"),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    prepocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return prepocessor
