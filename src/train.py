import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# Optional learners for stacking
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from .preprocess import build_preprocessor

RANDOM_STATE = 42

def train_pipeline(df: pd.DataFrame):
    # Expect raw (string) categorical columns as in README
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ('xgb', XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric='logloss', random_state=RANDOM_STATE)),
        ('lgbm', LGBMClassifier(n_estimators=400, random_state=RANDOM_STATE)),
        ('cb', CatBoostClassifier(iterations=400, learning_rate=0.05, verbose=0, random_state=RANDOM_STATE)),
    ]

    clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        n_jobs=-1
    )

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', clf)])
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    return pipeline, (X_test, y_test, y_pred, y_prob), metrics

def save_reports(X_test, y_test, y_pred, y_prob, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Confusion Matrix
    fig_cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    fig_cm.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/heart.csv")
    parser.add_argument("--out", type=str, default="models/stacking_pipeline.joblib")
    parser.add_argument("--reports", type=str, default="reports")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if 'HeartDisease' not in df.columns:
        raise ValueError("Expected 'HeartDisease' column in the dataset.")

    pipeline, eval_bundle, metrics = train_pipeline(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)

    # Save metrics
    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save plots
    X_test, y_test, y_pred, y_prob = eval_bundle
    save_reports(X_test, y_test, y_pred, y_prob, reports_dir)

    print(f"Model saved to: {out_path}")
    print(f"Reports saved to: {reports_dir}")

if __name__ == "__main__":
    main()
