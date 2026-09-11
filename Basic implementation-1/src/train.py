import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.features import FEATURE_COLUMNS

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"

def train_model():
    """
    Train Random Forest classifier on dataset and evaluate on test set.
    """
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError("Training or testing dataset not found. Please run prepare_dataset.py first.")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Validate columns
    for col in FEATURE_COLUMNS + ["label"]:
        if col not in train_df.columns:
            raise ValueError(f"Required column '{col}' missing from train dataset.")
        if col not in test_df.columns:
            raise ValueError(f"Required column '{col}' missing from test dataset.")

    # Validate classes
    unique_labels = train_df["label"].unique()
    if len(unique_labels) < 2:
        raise ValueError(f"Dataset must contain at least 2 target classes for training. Found: {unique_labels}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    print("========================================")
    print("      HireMinds AI - Model Training     ")
    print("========================================")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print("Initializing RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\nModel Performance Metrics (Test Set):")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"  FN: {cm[1][0]} | TP: {cm[1][1]}")

    print("\nFeature Importances:")
    importances = model.feature_importances_
    feature_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feature_imp:
        print(f"  - {feat:<22}: {imp * 100:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved successfully to: {MODEL_PATH}")
    print("========================================\n")

    return model, {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "feature_importances": dict(feature_imp)
    }

if __name__ == "__main__":
    train_model()
