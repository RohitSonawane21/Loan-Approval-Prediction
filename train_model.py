import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split


def load_and_prepare_data():
    data = pd.read_csv("loan_approval_dataset.csv")
    data.drop(columns=["loan_id"], inplace=True)
    data.columns = data.columns.str.strip()

    data["Assets"] = (
        data.residential_assets_value
        + data.commercial_assets_value
        + data.luxury_assets_value
        + data.bank_asset_value
    )
    data.drop(
        columns=[
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ],
        inplace=True,
    )

    data.education = data.education.str.strip().map({"Graduate": 1, "Not Graduate": 0}).astype(int)
    data.self_employed = data.self_employed.str.strip().map({"No": 0, "Yes": 1}).astype(int)
    data.loan_status = data.loan_status.str.strip().map({"Approved": 1, "Rejected": 0}).astype(int)

    input_data = data.drop(columns=["loan_status"])
    output_data = data["loan_status"]
    return train_test_split(input_data, output_data, test_size=0.2, random_state=42)


def train_model():
    x_train, x_test, y_train, y_test = load_and_prepare_data()

    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=1)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(best_model, "model.pkl")
    joblib.dump(accuracy, "model_accuracy.pkl")

    print("Best Parameters:", grid_search.best_params_)
    print("Tuned Accuracy:", accuracy)
    print("Saved model.pkl and model_accuracy.pkl")


if __name__ == "__main__":
    train_model()
