import joblib
import pandas as pd

from config import MODEL_PATH, TEST_DATA_PATH
from data import load_data
from features import add_features, categorize_features


def predict(input_data: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)

    input_data = categorize_features(input_data)
    input_data = add_features(input_data)

    predictions = model.predict(input_data)

    results = pd.DataFrame(
        {"Transported": predictions},
        index=input_data.index,
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)

        if probabilities.shape[1] == 2:
            results["positive_probability"] = probabilities[:, 1]

    return results


if __name__ == "__main__":
    new_data = pd.read_csv(TEST_DATA_PATH)

    results = predict(new_data)
    results["PassengerId"] = new_data["PassengerId"]

    results = results[["PassengerId", "Transported"]]

    results.to_csv(
        "reports/submission.csv",
        index=False,
    )
