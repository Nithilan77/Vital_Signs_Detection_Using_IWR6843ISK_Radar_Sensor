import os
import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
ROOT_DIR = r"d:\Nithilan\SEM 4\SPML\Vitals System Detection"
DATA_PATH = os.path.join(ROOT_DIR, "detectability_final.csv")
WORK_DIR = os.path.join(ROOT_DIR, "BodyPositionAnalysis")
MODEL_OUTPUT = os.path.join(WORK_DIR, "position_aware_hr_model.joblib")
METRICS_OUTPUT = os.path.join(WORK_DIR, "model_metrics.json")

def train_model():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found at {DATA_PATH}")
        return

    print(f"📌 Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Features and Target
    # We want to use the sensor's HR + Position features to predict the GROUND TRUTH HR
    categorical_features = ["Posture", "SensorPosition"]
    numeric_features = ["HeartRate_BPM", "Orientation_deg", "Distance_m", "SNR_dB", "WaveEnergy"]
    target = "GroundTruth_HR"

    # Drop rows where target or key features are NaN
    df = df.dropna(subset=[target] + numeric_features + categorical_features)

    X = df[numeric_features + categorical_features]
    y = df[target]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Full pipeline with XGBoost
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ))
    ])

    print("🚀 Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Baseline MAE (using raw sensor HR)
    baseline_mae = mean_absolute_error(y_test, X_test["HeartRate_BPM"])

    metrics = {
        "model_type": "XGBoost Regressor (Position-Aware)",
        "MAE": float(mae),
        "R2": float(r2),
        "Baseline_MAE_Raw_Sensor": float(baseline_mae),
        "Improvement_Percentage": float(((baseline_mae - mae) / baseline_mae) * 100)
    }

    print("\n📊 MODEL PERFORMANCE:")
    print(f"✅ Position-Aware MAE: {mae:.2f} BPM")
    print(f"📉 Baseline Sensor MAE: {baseline_mae:.2f} BPM")
    print(f"📈 Improvement: {metrics['Improvement_Percentage']:.1f}%")

    # Save
    joblib.dump(model, MODEL_OUTPUT)
    with open(METRICS_OUTPUT, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✔ Model saved to: {MODEL_OUTPUT}")
    print(f"✔ Metrics saved to: {METRICS_OUTPUT}")

if __name__ == "__main__":
    train_model()
