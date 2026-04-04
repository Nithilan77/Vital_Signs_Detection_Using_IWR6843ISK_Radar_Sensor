import os
import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
ROOT_DIR = r"d:\Nithilan\SEM 4\SPML\Vitals System Detection"
DATA_PATH = os.path.join(ROOT_DIR, "detectability_final.csv")
WORK_DIR = os.path.join(ROOT_DIR, "BodyPositionAnalysis")
MODEL_OUTPUT = os.path.join(WORK_DIR, "detectability_classifier.joblib")
METRICS_OUTPUT = os.path.join(WORK_DIR, "classifier_metrics.json")

def train_classifier():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found at {DATA_PATH}")
        return

    print(f"📌 Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Features and Target
    categorical_features = ["Posture", "SensorPosition"]
    numeric_features = ["Orientation_deg", "Distance_m", "SNR_dB", "WaveEnergy"]
    target = "Detected"

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

    # Full pipeline with XGBoost Classifier
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        ))
    ])

    print("🚀 Splitting data and training classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert keys to string for JSON serialization and robust access
    report = {str(k): v for k, v in report.items()}

    metrics = {
        "model_type": "XGBoost Classifier (Detectability)",
        "Accuracy": float(acc),
        "Classification_Report": report
    }

    print("\n📊 CLASSIFIER PERFORMANCE:")
    print(f"✅ Accuracy: {acc:.2%}")
    if '1.0' in report:
        print(f"✅ Precision (Class 1): {report['1.0']['precision']:.2%}")
        print(f"✅ Recall (Class 1): {report['1.0']['recall']:.2%}")
    elif '1' in report:
        print(f"✅ Precision (Class 1): {report['1']['precision']:.2%}")
        print(f"✅ Recall (Class 1): {report['1']['recall']:.2%}")

    # Save
    joblib.dump(model, MODEL_OUTPUT)
    with open(METRICS_OUTPUT, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✔ Classifier saved to: {MODEL_OUTPUT}")
    print(f"✔ Metrics saved to: {METRICS_OUTPUT}")

if __name__ == "__main__":
    train_classifier()
