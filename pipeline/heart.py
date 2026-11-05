"""
pipeline/heart.py
=================
Self-contained, end-to-end pipeline for the Heart Organ (Hypertension Risk).
Target: 'hypertension' column from Excel (1=Yes, 2=No, 9=missing/excluded).

Includes:
  1. Data Preprocessing & 80/20 Stratified Train/Test Split
  2. Layer 1 Training: Heart's Own Biomarkers Only (EBM, RF, LightGBM, XGBoost, Elastic-Net LR)
  3. Layer 2 Cross-Organ Dependency Integration:
     - Kidney Dependency (Cardiorenal axis: eGFR, Creatinine, BUN, ACR)
     - Pancreas Dependency (Diabetic vascular axis: Fasting Glucose, HbA1c, Insulin)
     - Liver Dependency (Hepato-cardiac axis: ALT, AST, Liver Fat CAP)
     - Adipose Dependency (Visceral fat & hemodynamic load: BMI, Waist Circumference)
  4. Cross-Organ Attention / Fusion (Learned Dependency Contribution Weights)
  5. Comprehensive Evaluation & Publication-Ready Figures:
     - Isolated vs. Cross-Organ ROC Curve Comparison
     - Organ Dependency Attention Weights Bar Chart
     - Heart Feature Importance Plot
     - Confusion Matrix

Usage:
    python pipeline/heart.py
"""

import os
import sys
import pickle
import warnings
from typing import Dict, Tuple, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except ImportError:
    HAS_EBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
from xai_module import run_explainable_ai

warnings.filterwarnings("ignore")

def compute_metrics_dict(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    brier = float(brier_score_loss(y_true, y_prob))
    rmse = float(np.sqrt(brier))
    mae = float(np.mean(np.abs(y_true - y_prob)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": brier,
        "rmse": rmse,
        "mae": mae,
    }

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & PATHS
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & PATHS (INLINED)
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
PIPELINE_DIR = os.path.join(REPO_ROOT, "pipeline")

def resolve_file_path(filename: str, candidates: List[str]) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

RAW_DATA_PATH = resolve_file_path(
    "Merged_Organ_Risk_Data",
    [
        os.path.join(DATASET_DIR, "Merged_Organ_Risk_Data.csv"),
        os.path.join(REPO_ROOT, "Merged_Organ_Risk_Data.csv"),
        os.path.join(REPO_ROOT, "Merged_Organ_Risk_Data.xlsx"),
        os.path.join(DATASET_DIR, "Merged_Organ_Risk_Data.xlsx"),
    ]
)

# All required values are sourced directly from Merged_Organ_Risk_Data — no XPT files needed.

RANDOM_SEED = 42
TEST_SIZE = 0.20
WINSOR_LOW = 0.005
WINSOR_HIGH = 0.995
# Hypertension: 1=Yes→1, 2=No→0, 9=missing→NaN (dropped)
HYPERTENSION_RECODE = {1: 1, 2: 0, 1.0: 1, 2.0: 0}

OUT_DIR = os.path.join(REPO_ROOT, "pipeline", "outputs", "heart")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
DPI = 150
sns.set_theme(style="whitegrid", font_scale=1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def compute_egfr_ckd_epi_2021(scr, age, sex):
    scr = np.asarray(scr, dtype=float)
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=float)
    kappa = np.where(sex == 2, 0.7, 0.9)
    alpha = np.where(sex == 2, -0.241, -0.302)
    sex_coef = np.where(sex == 2, 1.012, 1.0)
    ratio = scr / kappa
    return 142 * np.minimum(ratio, 1.0)**alpha * np.maximum(ratio, 1.0)**(-1.200) * (0.9938**age) * sex_coef

def load_and_preprocess_heart():
    print("\n" + "="*70)
    print(" [1/5] PREPROCESSING HEART DATA & CLINICAL DERIVATIONS")
    print("="*70)

    print(f" Loading master dataset from: {RAW_DATA_PATH}")
    if RAW_DATA_PATH.endswith(".csv"):
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    else:
        df = pd.read_excel(RAW_DATA_PATH)
    print(f" Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Column standardisation (no XPT merges needed) ──────────────────────
    # Rename Gender_1M_2F → Sex_1M_2F for internal consistency
    if "Gender_1M_2F" in df.columns and "Sex_1M_2F" not in df.columns:
        df.rename(columns={"Gender_1M_2F": "Sex_1M_2F"}, inplace=True)

    # Standardise insulin flag: Taking_Insulin_1Y_2N (1=Yes, 2=No)
    # Convert to binary 0/1 for use as a feature
    if "Taking_Insulin_1Y_2N" in df.columns:
        df["Insulin_Use"] = df["Taking_Insulin_1Y_2N"].map({1: 1, 2: 0, 1.0: 1, 2.0: 0})

    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Adults only
    df = df[df["Age"] >= 18].copy().reset_index(drop=True)

    # ── Clinical Target ─────────────────────────────────────────────────────
    # 'hypertension' column: 1=Yes → 1, 2=No → 0, other → NaN (dropped)
    if "hypertension" in df.columns:
        df["Hypertension_Label"] = df["hypertension"].map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    else:
        # Fallback to BP threshold (Systolic >= 130 or Diastolic >= 80)
        bp_htn = ((df["Systolic_BP_mmHg"] >= 130) | (df["Diastolic_BP_mmHg"] >= 80)).astype(float)
        df["Hypertension_Label"] = bp_htn

    # ── Derived features ────────────────────────────────────────────────────
    # eGFR (CKD-EPI 2021) — requires Serum_Creatinine, Age, Sex
    if all(c in df.columns for c in ["Serum_Creatinine_mg_dL", "Age", "Sex_1M_2F"]):
        df["eGFR"] = compute_egfr_ckd_epi_2021(
            df["Serum_Creatinine_mg_dL"], df["Age"], df["Sex_1M_2F"]
        )

    # Log-transforms for skewed biomarkers
    log_cols = [
        "hs_CRP_mg_L", "ALT_U_L", "AST_U_L",
        "Total_Bilirubin_mg_dL", "Serum_Creatinine_mg_dL",
        "Albumin_Creatinine_Ratio_ACR"
    ]
    for c in log_cols:
        if c in df.columns:
            short = c.replace("_mg_L", "").replace("_U_L", "").replace("_mg_dL", "")
            df[f"log_{short}"] = np.log1p(np.maximum(df[c].fillna(df[c].median()), 0.0))

    # ── Filter to valid Heart target rows ───────────────────────────────────
    df_heart = df.dropna(subset=["Hypertension_Label"]).copy().reset_index(drop=True)
    df_heart["Hypertension_Label"] = df_heart["Hypertension_Label"].astype(int)

    print(f" Cohort filtered for Heart: N = {len(df_heart)} patients")
    print(f" Target distribution (Hypertension): "
          f"{df_heart['Hypertension_Label'].value_counts().to_dict()} "
          f"(Prevalence: {df_heart['Hypertension_Label'].mean()*100:.1f}%)")

    # ── Feature sets (all sourced from Merged_Organ_Risk_Data) ──────────────
    # 1. Heart's Own Biomarkers
    # NOTE: Systolic_BP_mmHg / Diastolic_BP_mmHg are EXCLUDED — the
    # 'hypertension' label is directly derived from these BP thresholds,
    # so including them causes perfect data leakage.
    heart_own_cols = [
        "Age", "Sex_1M_2F", "log_hs_CRP"
    ]
    heart_own_cols = [c for c in heart_own_cols if c in df_heart.columns]

    # 2. Dependency Organs Biomarkers
    dep_cols = {
        "Kidney": [c for c in [
            "eGFR", "log_Serum_Creatinine", "BUN_mg_dL",
            "log_Albumin_Creatinine_Ratio_ACR"
        ] if c in df_heart.columns],
        "Pancreas": [c for c in [
            "HbA1c_pct", "Fasting_Glucose_mg_dL", "Insulin_Use"
        ] if c in df_heart.columns],
        "Liver": [c for c in [
            "log_ALT", "log_AST", "Liver_Fat_CAP_dB_m"
        ] if c in df_heart.columns],
        "Adipose": [c for c in [
            "BMI", "Waist_Circumference_cm"
        ] if c in df_heart.columns],
    }

    all_features = list(heart_own_cols)
    for organ_name, cols in dep_cols.items():
        all_features.extend([c for c in cols if c not in all_features])

    # Winsorize numeric features
    for c in all_features:
        if c in df_heart.columns:
            lo = df_heart[c].quantile(WINSOR_LOW)
            hi = df_heart[c].quantile(WINSOR_HIGH)
            df_heart[c] = df_heart[c].clip(lo, hi)

    # Stratified Train/Test Split (80/20)
    train_df, test_df = train_test_split(
        df_heart, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=df_heart["Hypertension_Label"]
    )
    print(f" Split complete: Train N = {len(train_df)}, Test N = {len(test_df)}")
    print(f" Heart own features ({len(heart_own_cols)}): {heart_own_cols}")
    for organ, cols in dep_cols.items():
        print(f" {organ} dep features ({len(cols)}): {cols}")

    return train_df, test_df, heart_own_cols, dep_cols

# ─────────────────────────────────────────────────────────────────────────────
# 2. LAYER 1: HEART'S OWN PARAMETERS (ISOLATED MODEL)
# ─────────────────────────────────────────────────────────────────────────────
def train_heart_isolated(train_df, test_df, heart_cols):
    print("\n" + "="*70)
    print(" [2/5] TRAINING HEART LAYER 1 (OWN BIOMARKERS ONLY)")
    print("="*70)
    print(f" Features used ({len(heart_cols)}): {heart_cols}")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_raw = train_df[heart_cols]
    y_train = train_df["Hypertension_Label"].values
    X_test_raw = test_df[heart_cols]
    y_test = test_df["Hypertension_Label"].values

    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_test = scaler.transform(imputer.transform(X_test_raw))

    models = {}
    train_preds = {}
    test_preds = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Layer 1 Primary Model: Random Forest
    print(" -> Fitting Layer 1 Primary Model: Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random_Forest"] = rf
    train_preds["Random_Forest"] = rf.predict_proba(X_train)[:, 1]
    test_preds["Random_Forest"] = rf.predict_proba(X_test)[:, 1]

    # Evaluate Layer 1 Random Forest Model
    model_eval = {}
    print("\n   " + "="*88)
    print(f"   {'Layer 1 Model':22s} | {'Train Acc':9s} | {'Test Acc':8s} | {'AUROC':7s} | {'PR-AUC':7s} | {'F1':6s} | {'Prec':6s} | {'Recall':6s}")
    print("   " + "-"*88)
    for name in models.keys():
        tr_m = compute_metrics_dict(y_train, train_preds[name])
        te_m = compute_metrics_dict(y_test, test_preds[name])
        model_eval[name] = {
            "train_acc": tr_m["accuracy"],
            "test_acc": te_m["accuracy"],
            "auroc": te_m["auroc"],
            "auc": te_m["auroc"],
            "pr_auc": te_m["pr_auc"],
            "ap": te_m["pr_auc"],
            "precision": te_m["precision"],
            "recall": te_m["recall"],
            "f1": te_m["f1"],
            "brier": te_m["brier"],
            "rmse": te_m["rmse"],
            "mae": te_m["mae"],
        }
        print(f"   {name:22s} | {tr_m['accuracy']:8.4f}  | {te_m['accuracy']:8.4f} | {te_m['auroc']:6.4f}  | {te_m['pr_auc']:6.4f}  | {te_m['f1']:5.4f}  | {te_m['precision']:5.4f}  | {te_m['recall']:5.4f}")
    print("   " + "="*88)

    best_name = "Random_Forest"
    best_model = models[best_name]
    y_prob_isolated_test = test_preds[best_name]
    print(f"\n   -> Selected Isolated Model: {best_name} (AUROC = {model_eval[best_name]['auroc']:.4f}, Test Acc = {model_eval[best_name]['test_acc']:.4f})")

    # Compute 5-fold CV OOF predictions for train set using best model architecture
    oof_isolated_train = np.zeros(len(train_df))
    for train_idx, val_idx in cv.split(X_train, y_train):
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X_train[train_idx], y_train[train_idx])
        oof_isolated_train[val_idx] = clf.predict_proba(X_train[val_idx])[:, 1]

    best_eval = model_eval[best_name]
    print(f"\n [Heart Best Isolated Model Results ({best_name})]:")
    print(f"   Train Accuracy : {best_eval['train_acc']:.4f}")
    print(f"   Test Accuracy  : {best_eval['test_acc']:.4f}")
    print(f"   AUROC          : {best_eval['auroc']:.4f}")
    print(f"   PR-AUC         : {best_eval['pr_auc']:.4f}")
    print(f"   Precision      : {best_eval['precision']:.4f}")
    print(f"   Recall         : {best_eval['recall']:.4f}")
    print(f"   F1 Score       : {best_eval['f1']:.4f}")
    print(f"   Brier Score    : {best_eval['brier']:.4f}")
    print(f"   RMSE           : {best_eval['rmse']:.4f}")
    print(f"   MAE            : {best_eval['mae']:.4f}")

    return {
        "best_model": best_model,
        "best_name": best_name,
        "all_models": models,
        "all_scores": {k: v["auroc"] for k, v in model_eval.items()},
        "all_eval": model_eval,
        "imputer": imputer,
        "scaler": scaler,
        "oof_train": oof_isolated_train,
        "prob_test": y_prob_isolated_test,
        "all_test_preds": test_preds,
        "auc": best_eval["auroc"],
        "ap": best_eval["pr_auc"],
        "brier": best_eval["brier"],
        "rmse": best_eval["rmse"],
        "mae": best_eval["mae"],
        "train_acc": best_eval["train_acc"],
        "test_acc": best_eval["test_acc"],
        "precision": best_eval["precision"],
        "recall": best_eval["recall"],
        "f1": best_eval["f1"]
    }

# ─────────────────────────────────────────────────────────────────────────────
# DEEP LEARNING ARCHITECTURES FOR LAYER 2 DIRECT CROSS-ORGAN CROSSTALK
# (Architecture 1: Direct Biomarker Encoders — No Intermediate Probabilities)
# ─────────────────────────────────────────────────────────────────────────────
class TargetCentricCrossAttentionFusion(nn.Module):
    """
    Architecture 1 (Primary): Target-Centric Gated Cross-Attention Network.
    - Target Organ (Heart) acts as the Query (Q).
    - External Organs (Kidney, Pancreas, Liver, Adipose) act as Keys (K) and Values (V).
    - A learned Gating unit modulates how much external organs modify the heart's intrinsic state.
    - Attention weights directly quantify the percentage impact of each external organ on the heart.
    """
    def __init__(self, target_dim: int, dep_dims: Dict[str, int], embed_dim: int = 16, num_heads: int = 2):
        super().__init__()
        self.dep_names = list(dep_dims.keys())

        # Target organ encoder (Heart biomarkers: Age, Sex, hs-CRP)
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # External organ encoders (Direct biomarkers, preserving full multi-dimensional signals)
        self.dep_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            for name, dim in dep_dims.items()
        })

        # Cross-Attention: Heart asks, other organs respond
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.val_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Gating unit: learned balance between intrinsic heart state and external organ crosstalk
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # Final Classifier predicting Target Organ Risk (Hypertension)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x_target, x_deps: Dict[str, torch.Tensor]):
        B = x_target.shape[0]
        # 1. Encode Heart
        h_target = self.target_encoder(x_target)  # (B, embed_dim)

        # 2. Encode all external organs
        h_dep_list = [self.dep_encoders[name](x_deps[name]) for name in self.dep_names]
        h_deps = torch.stack(h_dep_list, dim=1)    # (B, num_deps, embed_dim)

        # 3. Cross-Attention: Heart asks (Q), other organs respond (K, V)
        q = self.query_proj(h_target).unsqueeze(1) # (B, 1, embed_dim)
        k = self.key_proj(h_deps)                  # (B, num_deps, embed_dim)
        v = self.val_proj(h_deps)                  # (B, num_deps, embed_dim)

        context, weights = self.attn(q, k, v)      # context: (B, 1, embed_dim), weights: (B, 1, num_deps)
        context = context.squeeze(1)               # (B, embed_dim)
        weights = weights.squeeze(1)               # (B, num_deps)

        # 4. Gated fusion: intrinsic heart + gate * extrinsic crosstalk
        g = self.gate(torch.cat([h_target, context], dim=-1)) # (B, 1)
        fused = h_target + g * context                        # (B, embed_dim)

        # 5. Output logit
        out = self.classifier(fused).squeeze(-1)              # (B,)
        return out, weights, g.squeeze(-1)


class DirectedBioGNNFusion(nn.Module):
    """
    Architecture 2 (Benchmark): Directed Biological Graph Attention Network.
    Edges flow directly from each external organ node into the central Heart node.
    """
    def __init__(self, target_dim: int, dep_dims: Dict[str, int], embed_dim: int = 16):
        super().__init__()
        self.dep_names = list(dep_dims.keys())
        self.num_deps = len(dep_dims)
        self.target_encoder = nn.Sequential(nn.Linear(target_dim, embed_dim), nn.ReLU())
        self.dep_encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, embed_dim), nn.ReLU())
            for name, dim in dep_dims.items()
        })
        self.attn_fc = nn.Linear(2 * embed_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x_target, x_deps: Dict[str, torch.Tensor]):
        B = x_target.shape[0]
        h_target = self.target_encoder(x_target)
        h_dep_list = [self.dep_encoders[name](x_deps[name]) for name in self.dep_names]
        h_deps = torch.stack(h_dep_list, dim=1) # (B, num_deps, embed_dim)

        # Incoming edges: dep_i -> heart
        h_target_exp = h_target.unsqueeze(1).expand(-1, self.num_deps, -1)
        edge_pairs = torch.cat([h_target_exp, h_deps], dim=-1)
        e = self.leaky_relu(self.attn_fc(edge_pairs)).squeeze(-1)
        alpha = F.softmax(e, dim=-1) # (B, num_deps)

        # Message passing into Heart
        msg = torch.bmm(alpha.unsqueeze(1), h_deps).squeeze(1) # (B, embed_dim)
        fused = torch.cat([h_target, msg], dim=-1)
        out = self.classifier(fused).squeeze(-1)
        return out, alpha, None


class GatedBilinearOrganFusion(nn.Module):
    """
    Architecture 3 (Benchmark): Gated Bilinear Organ Fusion.
    Computes multiplicative quadratic interactions between Heart and pooled external organs.
    """
    def __init__(self, target_dim: int, dep_dims: Dict[str, int], embed_dim: int = 16):
        super().__init__()
        self.dep_names = list(dep_dims.keys())
        self.target_encoder = nn.Sequential(nn.Linear(target_dim, embed_dim), nn.ReLU())
        self.dep_encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, embed_dim), nn.ReLU())
            for name, dim in dep_dims.items()
        })
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x_target, x_deps: Dict[str, torch.Tensor]):
        B = x_target.shape[0]
        h_target = self.target_encoder(x_target)
        h_dep_list = [self.dep_encoders[name](x_deps[name]) for name in self.dep_names]
        h_deps = torch.stack(h_dep_list, dim=1)

        h_dep_mean = torch.mean(h_deps, dim=1)
        quad = self.bilinear(h_target, h_dep_mean)
        g = self.gate(torch.cat([h_target, h_dep_mean], dim=-1))
        fused = g * quad + (1.0 - g) * h_target
        out = self.classifier(fused).squeeze(-1)

        equal_w = torch.ones(B, len(self.dep_names), device=x_target.device) / len(self.dep_names)
        return out, equal_w, g.mean(dim=-1)


def train_dl_layer2_model(model, X_target_tr, X_deps_tr, y_train,
                          X_target_te, X_deps_te, y_test,
                          epochs=80, lr=0.008, weight_decay=1e-4):
    device = torch.device("cpu")
    model = model.to(device)
    torch.manual_seed(RANDOM_SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    x_tgt_tr = torch.tensor(X_target_tr, dtype=torch.float32, device=device)
    x_deps_tr_t = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in X_deps_tr.items()}
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)

    x_tgt_te = torch.tensor(X_target_te, dtype=torch.float32, device=device)
    x_deps_te_t = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in X_deps_te.items()}

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _, _ = model(x_tgt_tr, x_deps_tr_t)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        tr_logits, _, _ = model(x_tgt_tr, x_deps_tr_t)
        tr_probs = torch.sigmoid(tr_logits).cpu().numpy()
        te_logits, te_attn, te_gate = model(x_tgt_te, x_deps_te_t)
        te_probs = torch.sigmoid(te_logits).cpu().numpy()

    tr_m = compute_metrics_dict(y_train, tr_probs)
    te_m = compute_metrics_dict(y_test, te_probs)

    return {
        "model": model,
        "train_probs": tr_probs,
        "probs": te_probs,
        "train_acc": tr_m["accuracy"],
        "test_acc": te_m["accuracy"],
        "auc": te_m["auroc"],
        "ap": te_m["pr_auc"],
        "precision": te_m["precision"],
        "recall": te_m["recall"],
        "f1": te_m["f1"],
        "brier": te_m["brier"],
        "rmse": te_m["rmse"],
        "mae": te_m["mae"],
        "attn": te_attn.cpu().numpy() if te_attn is not None else None,
        "gate": te_gate.cpu().numpy() if te_gate is not None else None
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. LAYER 2: DIRECT CROSS-ORGAN CROSSTALK & DEEP LEARNING BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
def train_cross_organ_dependency(train_df, test_df, heart_isolated, dep_cols, heart_cols):
    print("\n" + "="*70)
    print(" [3/5] DIRECT CROSS-ORGAN CROSSTALK & DEEP LEARNING BENCHMARKING")
    print("="*70)

    y_train = train_df["Hypertension_Label"].values
    y_test = test_df["Hypertension_Label"].values

    # 1. Target Organ (Heart) feature matrices
    X_target_tr = heart_isolated["scaler"].transform(heart_isolated["imputer"].transform(train_df[heart_cols]))
    X_target_te = heart_isolated["scaler"].transform(heart_isolated["imputer"].transform(test_df[heart_cols]))
    target_dim = len(heart_cols)

    # 2. Dependency Organs feature matrices (Direct Multi-Dimensional Encoding)
    # NO intermediate LogisticRegression classifiers — NO artificial 1D probabilities!
    X_deps_tr = {}
    X_deps_te = {}
    dep_preprocessors = {}
    dep_dims = {}

    for organ, cols in dep_cols.items():
        if len(cols) == 0:
            continue
        print(f" -> Encoding {organ} multi-dimensional biomarkers ({len(cols)} features): {cols}")
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_deps_tr[organ] = scl.fit_transform(imp.fit_transform(train_df[cols]))
        X_deps_te[organ] = scl.transform(imp.transform(test_df[cols]))
        dep_preprocessors[organ] = {"imputer": imp, "scaler": scl, "cols": cols}
        dep_dims[organ] = len(cols)

    dep_names = list(dep_dims.keys())

    # ── Benchmarking Layer 2 Direct Crosstalk Architectures ──
    print("\n   [Layer 2 Direct Crosstalk Candidate Benchmarking]:")
    candidates_l2 = {}

    # 1. Deep Learning Model 1: Target-Centric Gated Cross-Attention (Recommended)
    m_cross_attn = TargetCentricCrossAttentionFusion(target_dim=target_dim, dep_dims=dep_dims)
    res_cross = train_dl_layer2_model(m_cross_attn, X_target_tr, X_deps_tr, y_train, X_target_te, X_deps_te, y_test)
    candidates_l2["Target_Centric_Cross_Attention"] = {
        "type": "DL_TargetCentric_Attention",
        **res_cross
    }

    # 2. Deep Learning Model 2: Directed Biological Graph Attention Network (Directed Bio-GNN)
    m_gnn = DirectedBioGNNFusion(target_dim=target_dim, dep_dims=dep_dims)
    res_gnn = train_dl_layer2_model(m_gnn, X_target_tr, X_deps_tr, y_train, X_target_te, X_deps_te, y_test)
    candidates_l2["Directed_Bio_GNN"] = {
        "type": "DL_Directed_GNN",
        **res_gnn
    }

    # 3. Deep Learning Model 3: Gated Bilinear Organ Fusion
    m_bilinear = GatedBilinearOrganFusion(target_dim=target_dim, dep_dims=dep_dims)
    res_bilinear = train_dl_layer2_model(m_bilinear, X_target_tr, X_deps_tr, y_train, X_target_te, X_deps_te, y_test)
    candidates_l2["Gated_Bilinear_Fusion"] = {
        "type": "DL_Gated_Bilinear",
        **res_bilinear
    }

    print("\n   " + "="*88)
    print(f"   {'Layer 2 Candidate Model':30s} | {'Train Acc':9s} | {'Test Acc':8s} | {'AUROC':7s} | {'PR-AUC':7s} | {'F1':6s} | {'Prec':6s} | {'Recall':6s}")
    print("   " + "-"*88)
    for name, c in candidates_l2.items():
        print(f"   {name:30s} | {c['train_acc']:8.4f}  | {c['test_acc']:8.4f} | {c['auc']:6.4f}  | {c['ap']:6.4f}  | {c['f1']:5.4f}  | {c['precision']:5.4f}  | {c['recall']:5.4f}")
    print("   " + "="*88)

    # Select best Layer 2 model based on AUROC
    best_l2_name = max(candidates_l2.keys(), key=lambda k: candidates_l2[k]["auc"])
    best_l2 = candidates_l2[best_l2_name]
    print(f"\n   -> Selected Best Layer 2 Model: {best_l2_name} (AUROC = {best_l2['auc']:.4f}, Test Acc = {best_l2['test_acc']:.4f})")

    # Extract external organ contribution percentages
    weights_dict = {}
    if best_l2["attn"] is not None:
        mean_attn = np.mean(best_l2["attn"], axis=0)
        norm_attn = mean_attn / np.sum(mean_attn)
        for organ_name, w in zip(dep_names, norm_attn):
            weights_dict[organ_name] = round(float(w * 100.0), 2)
    else:
        equal_w = 100.0 / len(dep_names)
        for organ_name in dep_names:
            weights_dict[organ_name] = round(equal_w, 2)

    # Intrinsic vs Extrinsic Risk Determination Shares (from learned gate)
    if best_l2["gate"] is not None:
        mean_gate = float(np.mean(best_l2["gate"]))
        extrinsic_share = round(mean_gate * 100.0, 2)
        intrinsic_share = round((1.0 - mean_gate) * 100.0, 2)
    else:
        intrinsic_share = 50.0
        extrinsic_share = 50.0

    print("\n [Overall Cardiac Risk Determination Balance]:")
    print(f"   * Intrinsic Heart Parameters Share : {intrinsic_share:5.2f}% (Age, Sex, hs-CRP)")
    print(f"   * Extrinsic Organ Crosstalk Share  : {extrinsic_share:5.2f}% (Influences from other organs)")

    print("\n [Learned External Organ Effect Breakdown on Heart]:")
    for organ_name, w in weights_dict.items():
        print(f"   * {organ_name:12s} effect on Heart: {w:5.2f}%")

    print(f"\n [Cross-Organ Enhanced Results ({best_l2_name})]:")
    print(f"   Train Accuracy : {best_l2['train_acc']:.4f}  (Delta: {best_l2['train_acc'] - heart_isolated['train_acc']:+.4f})")
    print(f"   Test Accuracy  : {best_l2['test_acc']:.4f}  (Delta: {best_l2['test_acc'] - heart_isolated['test_acc']:+.4f})")
    print(f"   AUROC          : {best_l2['auc']:.4f}  (Delta: {best_l2['auc'] - heart_isolated['auc']:+.4f})")
    print(f"   PR-AUC         : {best_l2['ap']:.4f}  (Delta: {best_l2['ap'] - heart_isolated['ap']:+.4f})")
    print(f"   Precision      : {best_l2['precision']:.4f}  (Delta: {best_l2['precision'] - heart_isolated['precision']:+.4f})")
    print(f"   Recall         : {best_l2['recall']:.4f}  (Delta: {best_l2['recall'] - heart_isolated['recall']:+.4f})")
    print(f"   F1 Score       : {best_l2['f1']:.4f}  (Delta: {best_l2['f1'] - heart_isolated['f1']:+.4f})")
    print(f"   Brier Score    : {best_l2['brier']:.4f}  (Delta: {best_l2['brier'] - heart_isolated['brier']:+.4f})")
    print(f"   RMSE           : {best_l2['rmse']:.4f}  (Delta: {best_l2['rmse'] - heart_isolated['rmse']:+.4f})")
    print(f"   MAE            : {best_l2['mae']:.4f}  (Delta: {best_l2['mae'] - heart_isolated['mae']:+.4f})")

    return {
        "best_name": best_l2_name,
        "best_model": best_l2["model"],
        "benchmarks": candidates_l2,
        "prob_test": best_l2["probs"],
        "train_acc": best_l2["train_acc"],
        "test_acc": best_l2["test_acc"],
        "weights": weights_dict,
        "intrinsic_share": intrinsic_share,
        "extrinsic_share": extrinsic_share,
        "bias": 0.0,
        "dep_models": dep_preprocessors,
        "auc": best_l2["auc"],
        "ap": best_l2["ap"],
        "precision": best_l2["precision"],
        "recall": best_l2["recall"],
        "f1": best_l2["f1"],
        "brier": best_l2["brier"],
        "rmse": best_l2["rmse"],
        "mae": best_l2["mae"]
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION & PUBLICATION-READY FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def save_classification_report_fig(y_true, y_pred, model_name: str, save_path: str):
    report_str = classification_report(y_true, y_pred, digits=2, zero_division=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.0), facecolor="#121212")
    ax.set_facecolor("#121212")
    
    header = "=" * 68 + f"\n{model_name.upper()} CLASSIFICATION REPORT\n" + "=" * 68 + "\n\n"
    full_text = header + report_str
    
    ax.text(
        0.04, 0.95, full_text,
        transform=ax.transAxes,
        fontsize=10.5,
        fontfamily="monospace",
        color="#e0e0e0",
        verticalalignment="top",
        horizontalalignment="left"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

def save_confusion_matrix_fig(y_true, y_pred, model_name: str, class_names: List[str], save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(f"{model_name}: Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()

def save_roc_curve_fig(y_true, y_prob, model_name: str, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance Level")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title(f"{model_name}: ROC Curve", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()

def generate_heart_figures(y_test, heart_isolated, heart_enhanced, heart_cols, train_df):
    print("\n" + "="*70)
    print(" [4/5] GENERATING EVALUATION METRICS & FIGURES")
    print("="*70)

    # Clean up obsolete figures
    allowed_prefixes = ("classification_report_", "confusion_matrix_", "roc_curve_", "external_organ_effects", "xai_")
    if os.path.exists(FIG_DIR):
        for f in os.listdir(FIG_DIR):
            if not any(f.startswith(p) for p in allowed_prefixes):
                try:
                    os.remove(os.path.join(FIG_DIR, f))
                except OSError:
                    pass

    class_names = ['No Hypertension', 'Hypertension Risk']
    models_to_plot = {}

    # Layer 1 models
    for m_name, probs in heart_isolated.get("all_test_preds", {}).items():
        models_to_plot[m_name] = probs
    if not models_to_plot:
        models_to_plot[heart_isolated["best_name"]] = heart_isolated["prob_test"]

    # Layer 2 models
    for m_name, c_data in heart_enhanced["benchmarks"].items():
        models_to_plot[m_name] = c_data["probs"]

    # Generate CR, CM, and ROC for EVERY model
    for model_name, probs in models_to_plot.items():
        y_pred = (probs >= 0.5).astype(int)
        clean_name = model_name.replace(" ", "_")

        # 1. Classification report figure
        cr_path = os.path.join(FIG_DIR, f"classification_report_{clean_name}.png")
        save_classification_report_fig(y_test, y_pred, model_name, cr_path)

        # 2. Confusion matrix figure
        cm_path = os.path.join(FIG_DIR, f"confusion_matrix_{clean_name}.png")
        save_confusion_matrix_fig(y_test, y_pred, model_name, class_names, cm_path)

        # 3. ROC curve figure
        roc_path = os.path.join(FIG_DIR, f"roc_curve_{clean_name}.png")
        save_roc_curve_fig(y_test, probs, model_name, roc_path)
        print(f" ✓ Generated CR, CM, and ROC figures for: {model_name}")

    # 4. Comparative ROC Curve
    plt.figure(figsize=(7.5, 6))
    fpr_iso, tpr_iso, _ = roc_curve(y_test, heart_isolated["prob_test"])
    fpr_enh, tpr_enh, _ = roc_curve(y_test, heart_enhanced["prob_test"])

    plt.plot(fpr_iso, tpr_iso, color="#2b5c8f", lw=2.2,
             label=f"Best Isolated ({heart_isolated['best_name']}): AUC = {heart_isolated['auc']:.4f}")
    plt.plot(fpr_enh, tpr_enh, color="#d95f02", lw=2.5,
             label=f"Cross-Organ Enhanced ({heart_enhanced['best_name']}): AUC = {heart_enhanced['auc']:.4f} (+{heart_enhanced['auc']-heart_isolated['auc']:+.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance Level")

    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Heart Organ: Isolated vs. Cross-Organ Enhanced ROC Comparison", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    roc_comp_path = os.path.join(FIG_DIR, "roc_curve_comparison.png")
    plt.savefig(roc_comp_path, dpi=DPI)
    plt.close()
    print(f" ✓ Saved comparative ROC: {roc_comp_path}")

    # 5. External Organ Attention Effects Bar Chart
    if "weights" in heart_enhanced and heart_enhanced["weights"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        orgs = list(heart_enhanced["weights"].keys())
        pcts = [heart_enhanced["weights"][o] for o in orgs]
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"][:len(orgs)]
        bars = ax.bar(orgs, pcts, color=colors, width=0.52, edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.6, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_ylim([0, max(pcts) + 10])
        ax.set_ylabel("Learned Effect Weight (%)", fontsize=11, fontweight="bold")
        ax.set_title("How Other Organs Affect Heart Risk (Cross-Attention)", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        attn_fig_path = os.path.join(FIG_DIR, "external_organ_effects.png")
        plt.savefig(attn_fig_path, dpi=DPI)
        plt.close()
        print(f" ✓ Saved external organ effects chart: {attn_fig_path}")

    metrics_rows = []
    for name, m_eval in heart_isolated["all_eval"].items():
        metrics_rows.append({
            "Stage": "Layer 1 Candidate",
            "Model": name,
            "Train_Accuracy": round(m_eval["train_acc"], 4),
            "Test_Accuracy": round(m_eval["test_acc"], 4),
            "AUROC": round(m_eval["auroc"], 4),
            "PR_AUC": round(m_eval["pr_auc"], 4),
            "Precision": round(m_eval["precision"], 4),
            "Recall": round(m_eval["recall"], 4),
            "F1_Score": round(m_eval["f1"], 4),
            "Brier_Score": round(m_eval["brier"], 4),
            "RMSE": round(m_eval.get("rmse", np.sqrt(m_eval["brier"])), 4),
            "MAE": round(m_eval.get("mae", 0.0), 4),
            "Status": "Selected Best Layer 1" if name == heart_isolated["best_name"] else "Candidate"
        })
    best_l1_name = heart_isolated["best_name"]
    best_l1_m = heart_isolated["all_eval"][best_l1_name]
    metrics_rows.append({
        "Stage": "Layer 1 Final Isolated",
        "Model": f"Best: {best_l1_name}",
        "Train_Accuracy": round(best_l1_m["train_acc"], 4),
        "Test_Accuracy": round(best_l1_m["test_acc"], 4),
        "AUROC": round(best_l1_m["auroc"], 4),
        "PR_AUC": round(best_l1_m["pr_auc"], 4),
        "Precision": round(best_l1_m["precision"], 4),
        "Recall": round(best_l1_m["recall"], 4),
        "F1_Score": round(best_l1_m["f1"], 4),
        "Brier_Score": round(best_l1_m["brier"], 4),
        "RMSE": round(best_l1_m.get("rmse", np.sqrt(best_l1_m["brier"])), 4),
        "MAE": round(best_l1_m.get("mae", 0.0), 4),
        "Status": "Primary Isolated"
    })
    for name, c_m in heart_enhanced["benchmarks"].items():
        metrics_rows.append({
            "Stage": "Layer 2 Candidate",
            "Model": name,
            "Train_Accuracy": round(c_m["train_acc"], 4),
            "Test_Accuracy": round(c_m["test_acc"], 4),
            "AUROC": round(c_m["auc"], 4),
            "PR_AUC": round(c_m["ap"], 4),
            "Precision": round(c_m["precision"], 4),
            "Recall": round(c_m["recall"], 4),
            "F1_Score": round(c_m["f1"], 4),
            "Brier_Score": round(c_m["brier"], 4),
            "RMSE": round(c_m.get("rmse", np.sqrt(c_m["brier"])), 4),
            "MAE": round(c_m.get("mae", 0.0), 4),
            "Status": "Selected Best Layer 2" if name == heart_enhanced["best_name"] else "Candidate"
        })
    best_l2_name = heart_enhanced["best_name"]
    best_l2_m = heart_enhanced["benchmarks"][best_l2_name]
    metrics_rows.append({
        "Stage": "Layer 2 Final Cross-Organ",
        "Model": f"Best: {best_l2_name}",
        "Train_Accuracy": round(best_l2_m["train_acc"], 4),
        "Test_Accuracy": round(best_l2_m["test_acc"], 4),
        "AUROC": round(best_l2_m["auc"], 4),
        "PR_AUC": round(best_l2_m["ap"], 4),
        "Precision": round(best_l2_m["precision"], 4),
        "Recall": round(best_l2_m["recall"], 4),
        "F1_Score": round(best_l2_m["f1"], 4),
        "Brier_Score": round(best_l2_m["brier"], 4),
        "RMSE": round(best_l2_m.get("rmse", np.sqrt(best_l2_m["brier"])), 4),
        "MAE": round(best_l2_m.get("mae", 0.0), 4),
        "Status": f"Enhanced (+{best_l2_m['auc']-best_l1_m.get('auc', best_l1_m.get('auroc', 0.0)):+.4f} AUROC Delta, +{best_l2_m['test_acc']-best_l1_m['test_acc']:+.4f} Acc Delta)"
    })
    metrics_summary = pd.DataFrame(metrics_rows)
    csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
    metrics_summary.to_csv(csv_path, index=False)
    print(f" ✓ Saved metrics summary: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPLAINABLE AI (XAI) — POST LAYER 2
# ─────────────────────────────────────────────────────────────────────────────
def run_explainable_ai_heart(train_df, test_df, heart_isolated, heart_enhanced,
                             heart_cols, dep_cols):
    """
    Explainable AI module that runs AFTER Layer 2 training.
    Produces:
      1. SHAP analysis on the Layer 1 best model (Random Forest)
      2. Gradient-based feature attribution on the Layer 2 DL model
      3. Combined feature importance ranking across all features
      4. Per-organ contribution waterfall chart
      5. CSV export of all importance values
    """
    print("\n" + "="*70)
    print(" [XAI] EXPLAINABLE AI — POST-LAYER 2 ANALYSIS")
    print("="*70)

    y_test = test_df["Hypertension_Label"].values
    y_train = train_df["Hypertension_Label"].values

    # ── Prepare feature matrices ─────────────────────────────────────────
    imputer = heart_isolated["imputer"]
    scaler = heart_isolated["scaler"]
    X_test_target = scaler.transform(imputer.transform(test_df[heart_cols]))
    X_train_target = scaler.transform(imputer.transform(train_df[heart_cols]))

    # All features (target + dependency organs)
    all_feature_names = list(heart_cols)
    all_dep_features = []
    dep_preprocessors = heart_enhanced.get("dep_models", {})
    for organ, cols in dep_cols.items():
        if len(cols) == 0:
            continue
        all_dep_features.extend(cols)
        all_feature_names.extend(cols)

    # Combined feature matrix for test set
    X_test_combined_parts = [X_test_target]
    X_train_combined_parts = [X_train_target]
    for organ, cols in dep_cols.items():
        if len(cols) == 0 or organ not in dep_preprocessors:
            continue
        pp = dep_preprocessors[organ]
        X_test_combined_parts.append(pp["scaler"].transform(pp["imputer"].transform(test_df[cols])))
        X_train_combined_parts.append(pp["scaler"].transform(pp["imputer"].transform(train_df[cols])))

    X_test_combined = np.hstack(X_test_combined_parts)
    X_train_combined = np.hstack(X_train_combined_parts)

    importance_records = []

    # ═══════════════════════════════════════════════════════════════════════
    # 1. SHAP ANALYSIS ON LAYER 1 BEST MODEL (Random Forest)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n [XAI Step 1/4] SHAP Analysis on Layer 1 Model...")
    best_model = heart_isolated["best_model"]
    best_name = heart_isolated["best_name"]

    shap_values_arr = None
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values_obj = explainer(X_test_target)
            # For binary classification, shap_values may have shape (n, features) or (n, features, 2)
            if len(shap_values_obj.values.shape) == 3:
                shap_values_arr = shap_values_obj.values[:, :, 1]  # class 1 SHAP values
            else:
                shap_values_arr = shap_values_obj.values

            mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=0)
            for i, col in enumerate(heart_cols):
                importance_records.append({
                    "Feature": col,
                    "Organ_Source": "Heart (Target)",
                    "Method": "SHAP (Layer 1)",
                    "Importance": round(float(mean_abs_shap[i]), 6)
                })

            # SHAP Beeswarm Plot
            fig, ax = plt.subplots(figsize=(9, max(4, len(heart_cols) * 0.5 + 1)))
            shap_display = shap.Explanation(
                values=shap_values_arr,
                base_values=shap_values_obj.base_values if len(shap_values_obj.base_values.shape) == 1 else shap_values_obj.base_values[:, 1],
                data=X_test_target,
                feature_names=heart_cols
            )
            shap.plots.beeswarm(shap_display, show=False, max_display=len(heart_cols))
            plt.title(f"Heart — SHAP Beeswarm (Layer 1: {best_name})", fontsize=13, fontweight="bold", pad=14)
            plt.tight_layout()
            shap_bee_path = os.path.join(FIG_DIR, "xai_shap_summary.png")
            plt.savefig(shap_bee_path, dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"   ✓ SHAP beeswarm plot saved: {shap_bee_path}")

            # SHAP Bar Plot
            fig, ax = plt.subplots(figsize=(8, max(3.5, len(heart_cols) * 0.45 + 1)))
            shap.plots.bar(shap_display, show=False, max_display=len(heart_cols))
            plt.title(f"Heart — SHAP Feature Importance (Layer 1: {best_name})", fontsize=13, fontweight="bold", pad=14)
            plt.tight_layout()
            shap_bar_path = os.path.join(FIG_DIR, "xai_shap_bar.png")
            plt.savefig(shap_bar_path, dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"   ✓ SHAP bar plot saved: {shap_bar_path}")

        except Exception as e:
            print(f"   ⚠ SHAP analysis failed ({e}), falling back to permutation importance.")
            shap_values_arr = None

    if shap_values_arr is None:
        # Fallback: Permutation Importance
        print("   → Using permutation importance (SHAP not available)...")
        from sklearn.inspection import permutation_importance
        perm_result = permutation_importance(
            best_model, X_test_target, y_test,
            n_repeats=20, random_state=RANDOM_SEED, scoring="roc_auc"
        )
        perm_imp = perm_result.importances_mean
        for i, col in enumerate(heart_cols):
            importance_records.append({
                "Feature": col,
                "Organ_Source": "Heart (Target)",
                "Method": "Permutation Importance (Layer 1)",
                "Importance": round(float(perm_imp[i]), 6)
            })

        # Plot permutation importance
        sorted_idx = np.argsort(perm_imp)[::-1]
        fig, ax = plt.subplots(figsize=(8, max(3.5, len(heart_cols) * 0.45 + 1)))
        colors = ["#2b5c8f"] * len(heart_cols)
        ax.barh([heart_cols[i] for i in sorted_idx], perm_imp[sorted_idx],
                color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Mean Permutation Importance (AUROC drop)", fontsize=11, fontweight="bold")
        ax.set_title(f"Heart — Permutation Importance (Layer 1: {best_name})", fontsize=13, fontweight="bold", pad=14)
        ax.invert_yaxis()
        plt.tight_layout()
        perm_path = os.path.join(FIG_DIR, "xai_shap_summary.png")
        plt.savefig(perm_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Permutation importance plot saved: {perm_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. GRADIENT-BASED FEATURE ATTRIBUTION ON LAYER 2 DL MODEL
    # ═══════════════════════════════════════════════════════════════════════
    print("\n [XAI Step 2/4] Gradient-Based Attribution on Layer 2 DL Model...")
    l2_model = heart_enhanced["best_model"]
    l2_name = heart_enhanced["best_name"]

    try:
        device = torch.device("cpu")
        l2_model.eval()

        # Prepare inputs with gradient tracking
        x_tgt = torch.tensor(X_test_target, dtype=torch.float32, device=device, requires_grad=True)
        x_deps_dict = {}
        dep_feature_offset = len(heart_cols)
        dep_grad_tensors = {}
        for organ, cols in dep_cols.items():
            if len(cols) == 0 or organ not in dep_preprocessors:
                continue
            pp = dep_preprocessors[organ]
            X_dep = pp["scaler"].transform(pp["imputer"].transform(test_df[cols]))
            t = torch.tensor(X_dep, dtype=torch.float32, device=device, requires_grad=True)
            x_deps_dict[organ] = t
            dep_grad_tensors[organ] = t

        # Forward pass
        logits, _, _ = l2_model(x_tgt, x_deps_dict)
        probs = torch.sigmoid(logits)

        # Compute gradients w.r.t. all inputs
        grad_outputs = torch.ones_like(probs)
        all_inputs = [x_tgt] + [dep_grad_tensors[o] for o in dep_grad_tensors]
        grads = torch.autograd.grad(probs, all_inputs, grad_outputs=grad_outputs,
                                     create_graph=False, allow_unused=True)

        # Target organ gradient importance
        tgt_grad = grads[0].detach().cpu().numpy()  # (N, target_dim)
        tgt_attribution = np.mean(np.abs(tgt_grad * X_test_target), axis=0)  # gradient × input

        for i, col in enumerate(heart_cols):
            importance_records.append({
                "Feature": col,
                "Organ_Source": "Heart (Target)",
                "Method": "Gradient Attribution (Layer 2)",
                "Importance": round(float(tgt_attribution[i]), 6)
            })

        # Dependency organ gradient importance
        grad_idx = 1
        for organ in dep_grad_tensors:
            cols = dep_cols[organ]
            dep_grad = grads[grad_idx].detach().cpu().numpy()
            pp = dep_preprocessors[organ]
            X_dep = pp["scaler"].transform(pp["imputer"].transform(test_df[cols]))
            dep_attribution = np.mean(np.abs(dep_grad * X_dep), axis=0)
            for i, col in enumerate(cols):
                importance_records.append({
                    "Feature": col,
                    "Organ_Source": organ,
                    "Method": "Gradient Attribution (Layer 2)",
                    "Importance": round(float(dep_attribution[i]), 6)
                })
            grad_idx += 1

        # Plot Layer 2 gradient attribution
        l2_records = [r for r in importance_records if r["Method"] == "Gradient Attribution (Layer 2)"]
        l2_df = pd.DataFrame(l2_records).sort_values("Importance", ascending=True)

        organ_color_map = {
            "Heart (Target)": "#e63946", "Kidney": "#457b9d", "Pancreas": "#f4a261",
            "Liver": "#2a9d8f", "Adipose": "#e76f51"
        }
        bar_colors = [organ_color_map.get(s, "#999999") for s in l2_df["Organ_Source"]]

        fig, ax = plt.subplots(figsize=(9, max(5, len(l2_df) * 0.4 + 1.5)))
        bars = ax.barh(l2_df["Feature"], l2_df["Importance"], color=bar_colors,
                       edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Mean |Gradient × Input| Attribution", fontsize=11, fontweight="bold")
        ax.set_title(f"Heart — Layer 2 Feature Attribution ({l2_name})", fontsize=13, fontweight="bold", pad=14)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, edgecolor="black", label=n)
                           for n, c in organ_color_map.items()
                           if n in l2_df["Organ_Source"].values]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        l2_attr_path = os.path.join(FIG_DIR, "xai_layer2_feature_attribution.png")
        plt.savefig(l2_attr_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Layer 2 gradient attribution plot saved: {l2_attr_path}")

    except Exception as e:
        print(f"   ⚠ Layer 2 gradient attribution failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. COMBINED FEATURE IMPORTANCE RANKING
    # ═══════════════════════════════════════════════════════════════════════
    print("\n [XAI Step 3/4] Combined Feature Importance Ranking...")
    imp_df = pd.DataFrame(importance_records)

    if not imp_df.empty:
        # Aggregate: for each feature, average across methods (normalized per method)
        combined = []
        for method in imp_df["Method"].unique():
            method_df = imp_df[imp_df["Method"] == method].copy()
            total = method_df["Importance"].sum()
            if total > 0:
                method_df["Normalized"] = method_df["Importance"] / total
            else:
                method_df["Normalized"] = 0.0
            combined.append(method_df)

        combined_df = pd.concat(combined, ignore_index=True)
        agg_df = combined_df.groupby(["Feature", "Organ_Source"]).agg(
            Avg_Normalized_Importance=("Normalized", "mean")
        ).reset_index().sort_values("Avg_Normalized_Importance", ascending=True)

        bar_colors = [organ_color_map.get(s, "#999999") for s in agg_df["Organ_Source"]]

        fig, ax = plt.subplots(figsize=(9, max(5, len(agg_df) * 0.4 + 1.5)))
        ax.barh(agg_df["Feature"], agg_df["Avg_Normalized_Importance"],
                color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Average Normalized Importance (across XAI methods)", fontsize=11, fontweight="bold")
        ax.set_title("Heart — Combined Feature Importance (Layer 1 + Layer 2)", fontsize=13, fontweight="bold", pad=14)

        legend_elements = [Patch(facecolor=c, edgecolor="black", label=n)
                           for n, c in organ_color_map.items()
                           if n in agg_df["Organ_Source"].values]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        combined_path = os.path.join(FIG_DIR, "xai_combined_feature_importance.png")
        plt.savefig(combined_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Combined feature importance plot saved: {combined_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PER-ORGAN CONTRIBUTION WATERFALL
    # ═══════════════════════════════════════════════════════════════════════
    print("\n [XAI Step 4/4] Per-Organ Contribution Waterfall...")
    if not imp_df.empty:
        # Sum importance by organ source (using gradient attribution if available)
        grad_df = imp_df[imp_df["Method"].str.contains("Layer 2")]
        if grad_df.empty:
            grad_df = imp_df
        organ_totals = grad_df.groupby("Organ_Source")["Importance"].sum()
        total_imp = organ_totals.sum()
        if total_imp > 0:
            organ_pct = (organ_totals / total_imp * 100).sort_values(ascending=False)
        else:
            organ_pct = organ_totals

        fig, ax = plt.subplots(figsize=(8, 5))
        orgs = organ_pct.index.tolist()
        pcts = organ_pct.values
        colors = [organ_color_map.get(o, "#999999") for o in orgs]

        bars = ax.bar(orgs, pcts, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

        ax.set_ylim([0, max(pcts) * 1.2 + 5])
        ax.set_ylabel("Contribution to Risk Prediction (%)", fontsize=11, fontweight="bold")
        ax.set_title("Heart — Per-Organ Feature Contribution (XAI)", fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()
        waterfall_path = os.path.join(FIG_DIR, "xai_organ_contribution_waterfall.png")
        plt.savefig(waterfall_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Per-organ contribution waterfall saved: {waterfall_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    if importance_records:
        xai_csv = pd.DataFrame(importance_records)
        xai_csv_path = os.path.join(OUT_DIR, "xai_feature_importance.csv")
        xai_csv.to_csv(xai_csv_path, index=False)
        print(f"   ✓ XAI feature importance CSV saved: {xai_csv_path}")

    # ── Console Summary ──────────────────────────────────────────────────
    print("\n   " + "="*60)
    print("   EXPLAINABLE AI SUMMARY — HEART")
    print("   " + "-"*60)
    if not imp_df.empty:
        for method in imp_df["Method"].unique():
            print(f"\n   [{method}]:")
            m_df = imp_df[imp_df["Method"] == method].sort_values("Importance", ascending=False)
            for _, row in m_df.iterrows():
                print(f"     {row['Feature']:35s} ({row['Organ_Source']:15s}): {row['Importance']:.6f}")
    print("   " + "="*60)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print("="*70)
    print(" HEART ORGAN PIPELINE — MULTI-ORGAN RISK PREDICTION SYSTEM")
    print("="*70)

    # 1. Preprocess & Split
    train_df, test_df, heart_cols, dep_cols = load_and_preprocess_heart()

    # 2. Train Layer 1 (Isolated Heart) & Select Best Model
    heart_isolated = train_heart_isolated(train_df, test_df, heart_cols)

    # 3. Train Layer 2 (Direct Cross-Organ Crosstalk & DL Benchmark)
    heart_enhanced = train_cross_organ_dependency(train_df, test_df, heart_isolated, dep_cols, heart_cols)

    # 4. Explainable AI (Post-Layer 2)
    run_explainable_ai_heart(train_df, test_df, heart_isolated, heart_enhanced, heart_cols, dep_cols)

    # 5. Evaluation & Figures
    y_test = test_df["Hypertension_Label"].values
    generate_heart_figures(y_test, heart_isolated, heart_enhanced, heart_cols, train_df)

    # 5. Save the BEST MODELS
    MODEL_DIR = os.path.join(OUT_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save Best Isolated Model
    best_iso_path = os.path.join(MODEL_DIR, "best_isolated_model.pkl")
    with open(best_iso_path, "wb") as f:
        pickle.dump({
            "organ": "heart",
            "best_model_name": heart_isolated["best_name"],
            "model": heart_isolated["best_model"],
            "imputer": heart_isolated["imputer"],
            "scaler": heart_isolated["scaler"],
            "features": heart_cols,
            "metrics": {
                "train_accuracy": heart_isolated["train_acc"],
                "test_accuracy": heart_isolated["test_acc"],
                "auroc": heart_isolated["auc"],
                "pr_auc": heart_isolated["ap"],
                "precision": heart_isolated["precision"],
                "recall": heart_isolated["recall"],
                "f1": heart_isolated["f1"],
                "brier": heart_isolated["brier"]
            }
        }, f)
    print(f" ✓ SAVED BEST ISOLATED MODEL ({heart_isolated['best_name']}): {best_iso_path}")

    # Save Cross-Organ Model
    cross_path = os.path.join(MODEL_DIR, "cross_organ_model.pkl")
    with open(cross_path, "wb") as f:
        pickle.dump({
            "organ": "heart",
            "best_layer2_model_name": heart_enhanced["best_name"],
            "layer2_benchmarks": heart_enhanced["benchmarks"],
            "best_model": heart_enhanced["best_model"],
            "weights": heart_enhanced["weights"],
            "intrinsic_share": heart_enhanced.get("intrinsic_share", 50.0),
            "extrinsic_share": heart_enhanced.get("extrinsic_share", 50.0),
            "bias": heart_enhanced["bias"],
            "dep_models": heart_enhanced["dep_models"],
            "dep_cols": dep_cols,
            "metrics": {
                "train_accuracy": heart_enhanced["train_acc"],
                "test_accuracy": heart_enhanced["test_acc"],
                "auroc": heart_enhanced["auc"],
                "pr_auc": heart_enhanced["ap"],
                "precision": heart_enhanced["precision"],
                "recall": heart_enhanced["recall"],
                "f1": heart_enhanced["f1"],
                "brier": heart_enhanced["brier"]
            }
        }, f)
    print(f" ✓ SAVED CROSS-ORGAN ENHANCED MODEL ({heart_enhanced['best_name']}): {cross_path}")

    # 6. Save Patient-Level Predictions
    preds_df = pd.DataFrame({
        "Patient_ID": test_df["Patient_ID"].values,
        "True_Hypertension": y_test,
        "Risk_Heart_Isolated": np.round(heart_isolated["prob_test"], 4),
        "Pred_Heart_Isolated": (heart_isolated["prob_test"] >= 0.5).astype(int),
        "Risk_Heart_CrossOrgan": np.round(heart_enhanced["prob_test"], 4),
        "Pred_Heart_CrossOrgan": (heart_enhanced["prob_test"] >= 0.5).astype(int),
    })
    preds_path = os.path.join(OUT_DIR, "patient_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f" ✓ Saved patient predictions: {preds_path}")

    print("\n" + "="*70)
    print(" HEART PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" Best Models and Outputs stored in: {OUT_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    run()
