from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_safe = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)))


def load_data(project_root: Path) -> pd.DataFrame:
    data_dir = project_root / "data"
    ridership = pd.read_csv(data_dir / "ridership.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")

    ridership["date"] = pd.to_datetime(ridership["date"])  # normalize dtype
    calendar["date"] = pd.to_datetime(calendar["date"])  # normalize dtype

    df = ridership.merge(calendar, on="date", how="left")
    # Ensure columns exist
    for col in ["day_of_week", "is_holiday", "month", "weather_index"]:
        if col not in df.columns:
            raise ValueError(f"Missing column after merge: {col}")
    return df


def train_and_save(project_root: Path, test_days: int, random_state: int) -> None:
    df = load_data(project_root)

    # Features and target
    df["hour"] = df["hour"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["month"] = df["month"].astype(int)
    df["weather_index"] = df["weather_index"].astype(float)

    feature_cols_cat = ["route_id", "stop_id"]
    feature_cols_num = ["hour", "day_of_week", "is_holiday", "month", "weather_index"]
    target_col = "passengers_count"

    df = df.sort_values(["date", "hour"]).reset_index(drop=True)

    # Time-based split: last N days as test
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    X_train = train_df[feature_cols_cat + feature_cols_num]
    y_train = train_df[target_col].astype(float).values
    X_test = test_df[feature_cols_cat + feature_cols_num]
    y_test = test_df[target_col].astype(float).values

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_cols_cat),
            ("num", "passthrough", feature_cols_num),
        ]
    )

    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.08,
        max_iter=300,
        l2_regularization=0.0,
        random_state=random_state,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", model),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE:  {mae:.2f}")
    print(f"Test MAPE: {mape*100:.2f}%")

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "demand_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train demand forecasting model for TNSTC ridership")
    parser.add_argument("--test_days", type=int, default=14)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_and_save(project_root, test_days=args.test_days, random_state=args.random_state)


if __name__ == "__main__":
    main()