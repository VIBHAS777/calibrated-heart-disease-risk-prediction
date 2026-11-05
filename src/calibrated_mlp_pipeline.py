"""
Full End-to-End Calibrated Deep Learning & Machine Learning Pipeline for Heart Disease Risk Prediction.
Publication: Engineered Science (ES-2026-0082)
Architecture:
  - Feature Engineering & Outlier Clipping
  - Interaction Terms (Cholesterol x BMI, BP Ratio, Age x Stress)
  - Class Imbalance Mitigation (RandomOverSampler)
  - Bagged Multi-Layer Perceptron (MLP) Neural Network
  - Dual Probabilistic Calibration: Sigmoid (Platt Scaling) + Isotonic Regression
  - Strict Clinical Reliability Scoring: Brier Score, Log-Loss, ROC-AUC
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, classification_report, roc_curve, auc

def balance_classes_oversample(X, y, random_state=42):
    """Native Random Over-Sampling for class balance."""
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y.values == 1)[0]
    neg_idx = np.where(y.values == 0)[0]
    n_target = max(len(pos_idx), len(neg_idx))
    pos_resampled = rng.choice(pos_idx, size=n_target, replace=True)
    neg_resampled = rng.choice(neg_idx, size=n_target, replace=True)
    res_idx = np.concatenate([pos_resampled, neg_resampled])
    rng.shuffle(res_idx)
    return X.iloc[res_idx].reset_index(drop=True), y.iloc[res_idx].reset_index(drop=True)

def find_dataset(filepath: str = None) -> str:
    if filepath and os.path.exists(filepath):
        return filepath
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "heart_attack_prediction_dataset.csv"),
        "data/heart_attack_prediction_dataset.csv",
        "heart_attack_prediction_dataset.csv",
        "/Users/vibhas/Documents/probup/heart_attack_prediction_dataset.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not locate heart_attack_prediction_dataset.csv")

def clip_outliers(X):
    X = X.copy()
    for col, (low, high) in {
        'Cholesterol': (100, 500),
        'Systolic_BP': (80, 220),
        'Diastolic_BP': (50, 140),
        'Heart Rate': (30, 200),
        'BMI': (10, 60),
        'Triglycerides': (30, 1000),
    }.items():
        if col in X:
            X[col] = X[col].clip(low, high)
    return X

def add_interaction_features(X):
    X = X.copy()
    if 'Cholesterol' in X and 'BMI' in X:
        X['Cholesterol_BMI'] = X['Cholesterol'] * X['BMI']
    if 'Systolic_BP' in X and 'Diastolic_BP' in X:
        X['BP_Ratio'] = X['Systolic_BP'] / (X['Diastolic_BP'] + 1)
    if 'Age' in X and 'Stress Level' in X:
        X['Age_Stress'] = X['Age'] * X['Stress Level']
    return X

def feature_engineering(X):
    X = clip_outliers(X)
    X = add_interaction_features(X)
    return X

def run_calibrated_pipeline(data_path: str = None, output_dir: str = "outputs") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    csv_file = find_dataset(data_path)
    print(f"[*] Loading Heart Attack Clinical Dataset: {csv_file}")
    df = pd.read_csv(csv_file)

    if 'Blood Pressure' in df.columns:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        df = df.drop(columns=["Blood Pressure"])
    if 'Patient ID' in df.columns:
        df = df.drop(columns=["Patient ID"])

    target = "Heart Attack Risk"
    categorical = [col for col in df.columns if df[col].dtype == 'object' and col != target]
    numerical = [col for col in df.columns if col not in categorical + [target]]

    # Feature transformation
    feature_engineer = FunctionTransformer(feature_engineering)
    X = df.drop(columns=[target])
    y = df[target]
    X = feature_engineer.fit_transform(X)

    numerical_extended = numerical.copy()
    for f in ['Cholesterol_BMI', 'BP_Ratio', 'Age_Stress']:
        if f in X.columns:
            numerical_extended.append(f)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Oversampling for class balance
    X_train_res, y_train_res = balance_classes_oversample(X_train, y_train, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_extended),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
    ])

    print("[*] Training Base Bagged MLP Neural Network Ensemble...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=300,
        alpha=0.0005,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        batch_size=128,
        learning_rate_init=0.001
    )

    bagged_mlp = BaggingClassifier(
        estimator=mlp,
        n_estimators=3,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    # Probability calibration (Platt Scaling & Isotonic Regression)
    print("[*] Calibrating Probabilities (Sigmoid Platt Scaling & Isotonic CV)...")
    calibrated_sigmoid = CalibratedClassifierCV(bagged_mlp, method='sigmoid', cv=3)
    calibrated_isotonic = CalibratedClassifierCV(bagged_mlp, method='isotonic', cv=3)

    pipe_sigmoid = Pipeline([("prep", preprocessor), ("clf", calibrated_sigmoid)])
    pipe_isotonic = Pipeline([("prep", preprocessor), ("clf", calibrated_isotonic)])

    pipe_sigmoid.fit(X_train_res, y_train_res)
    pipe_isotonic.fit(X_train_res, y_train_res)

    # Uncalibrated baseline model for comparison
    pipe_uncalibrated = Pipeline([("prep", preprocessor), ("clf", bagged_mlp)])
    pipe_uncalibrated.fit(X_train_res, y_train_res)

    p_uncalibrated = pipe_uncalibrated.predict_proba(X_test)[:, 1]
    p_sig = pipe_sigmoid.predict_proba(X_test)[:, 1]
    p_iso = pipe_isotonic.predict_proba(X_test)[:, 1]
    p_ens = (p_sig + p_iso) / 2.0

    y_pred = (p_ens > 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    raw_brier = float(brier_score_loss(y_test, p_uncalibrated))
    cal_brier = float(brier_score_loss(y_test, p_ens))
    logl = float(log_loss(y_test, p_ens))

    fpr, tpr, _ = roc_curve(y_test, p_ens)
    roc_auc = float(auc(fpr, tpr))

    # Save Calibration Curve Plot
    prob_true, prob_pred = calibration_curve(y_test, p_ens, n_bins=10)
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Dual Calibrated Ensemble')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Observed Fraction of Positives')
    plt.title('Clinical Risk Calibration Curve')
    plt.legend()
    plt.grid(True)
    cal_plot_path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(cal_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save ROC Curve Plot
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f'Dual Calibrated ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Heart Disease Prediction')
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_plot_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "test_samples": len(y_test),
        "test_accuracy": acc,
        "uncalibrated_brier_score": raw_brier,
        "calibrated_brier_score": cal_brier,
        "brier_improvement_percent": round(((raw_brier - cal_brier) / raw_brier) * 100, 2),
        "log_loss": logl,
        "roc_auc": roc_auc,
        "calibration_plot": cal_plot_path,
        "roc_plot": roc_plot_path
    }

if __name__ == "__main__":
    res = run_calibrated_pipeline()
    print("\nPipeline Complete:")
    print(res)
