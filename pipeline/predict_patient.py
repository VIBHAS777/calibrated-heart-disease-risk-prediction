"""
pipeline/predict_patient.py
===========================
Interactive & CLI Patient Risk Prediction Module.

Workflow:
  1. Takes all clinical biomarker inputs from the patient (or loads default demo).
  2. Asks the user to choose an organ:
       [H] Heart      - Hypertension Risk
       [K] Kidney     - Albuminuria / CKD Risk
       [L] Liver      - MASLD / Hepatic Steatosis Risk
       [P] Pancreas   - Adenocarcinoma / Pancreatic Risk
       [A] Adipose    - Visceral Adiposity / Obesity Risk
       [ALL] Check All 5 Organs
  3. Produces:
       a) Risk score for Layer 1 (Target organ intrinsic parameters only)
       b) Risk score for Layer 2 (Target organ + All external organs crosstalk)
       c) Pairwise crosstalk risk scores: e.g. (Heart, Kidney), (Heart, Liver),
          (Heart, Pancreas), (Heart, Adipose)

Usage:
  Interactive Mode:
    python pipeline/predict_patient.py

  Demo Mode (Pre-populated sample patient):
    python pipeline/predict_patient.py --demo
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure pipeline directory is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(REPO_ROOT, "pipeline")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import heart
import kidney
import liver
import pancreas
import adipose

# Register PyTorch neural network classes into sys.modules['__main__'] for pickle deserialization
for mod in [heart, kidney, liver, pancreas, adipose]:
    for attr in dir(mod):
        if "Fusion" in attr or "Ensemble" in attr:
            setattr(sys.modules["__main__"], attr, getattr(mod, attr))


ORGAN_REGISTRY = {
    "heart": {
        "key": "heart",
        "code": "H",
        "name": "Heart",
        "endpoint": "Hypertension Risk",
        "description": "Hypertension & cardiovascular pressure risk",
        "dep_cols": {
            "Kidney": ["eGFR", "log_Serum_Creatinine", "BUN_mg_dL", "log_Albumin_Creatinine_Ratio_ACR"],
            "Pancreas": ["HbA1c_pct", "Fasting_Glucose_mg_dL", "Insulin_Use"],
            "Liver": ["log_ALT", "log_AST", "Liver_Fat_CAP_dB_m"],
            "Adipose": ["BMI", "Waist_Circumference_cm"]
        }
    },
    "kidney": {
        "key": "kidney",
        "code": "K",
        "name": "Kidney",
        "endpoint": "Albuminuria / CKD Risk",
        "description": "Microvascular renal damage & filtration decline",
        "dep_cols": {
            "Heart": ["Systolic_BP_mmHg", "Diastolic_BP_mmHg"],
            "Pancreas": ["HbA1c_pct", "Fasting_Glucose_mg_dL", "Insulin_Use"],
            "Liver": ["log_ALT", "log_AST", "Liver_Fat_CAP_dB_m"],
            "Adipose": ["BMI", "Waist_Circumference_cm"]
        }
    },
    "liver": {
        "key": "liver",
        "code": "L",
        "name": "Liver",
        "endpoint": "MASLD / Steatosis Risk",
        "description": "Metabolic dysfunction-associated steatotic liver disease",
        "dep_cols": {
            "Pancreas": ["HbA1c_pct", "Fasting_Glucose_mg_dL", "Insulin_Use"],
            "Adipose": ["BMI", "Waist_Circumference_cm"],
            "Heart": ["Systolic_BP_mmHg", "Diastolic_BP_mmHg"],
            "Kidney": ["eGFR", "log_Serum_Creatinine", "BUN_mg_dL"]
        }
    },
    "pancreas": {
        "key": "pancreas",
        "code": "P",
        "name": "Pancreas",
        "endpoint": "Adenocarcinoma Risk",
        "description": "Pancreatic oncogenic & metabolic disease risk",
        "dep_cols": {
            "Adipose": ["BMI", "Waist_Circumference_cm"],
            "Liver": ["log_ALT", "log_AST", "Liver_Fat_CAP_dB_m"],
            "Kidney": ["eGFR", "log_Serum_Creatinine", "BUN_mg_dL", "log_Albumin_Creatinine_Ratio_ACR"],
            "Heart": ["Systolic_BP_mmHg", "Diastolic_BP_mmHg"]
        }
    },
    "adipose": {
        "key": "adipose",
        "code": "A",
        "name": "Adipose",
        "endpoint": "Visceral Adiposity Risk",
        "description": "Pathologic visceral adipose tissue accumulation",
        "dep_cols": {
            "Pancreas": ["HbA1c_pct", "Fasting_Glucose_mg_dL", "Insulin_Use"],
            "Liver": ["log_ALT", "log_AST", "Liver_Fat_CAP_dB_m"],
            "Heart": ["Systolic_BP_mmHg", "Diastolic_BP_mmHg"],
            "Kidney": ["eGFR", "log_Serum_Creatinine", "BUN_mg_dL"]
        }
    }
}


def compute_risk_tier(prob: float) -> str:
    """Returns clinical risk categorization."""
    if prob < 0.30:
        return "LOW RISK (Baseline / Normal)"
    elif prob < 0.60:
        return "MODERATE RISK (Borderline Elevated)"
    else:
        return "HIGH RISK (Significant Clinical Concern)"


def render_progress_bar(prob: float, width: int = 24) -> str:
    """Renders a clean ASCII terminal progress bar."""
    filled = max(0, min(width, int(round(prob * width))))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}] {prob * 100:5.1f}%"


def load_organ_models(organ_key: str):
    """Loads pre-trained Layer 1 and Layer 2 models for a specified organ."""
    organ_dir = os.path.join(PIPELINE_DIR, "outputs", organ_key.lower(), "models")
    iso_path = os.path.join(organ_dir, "best_isolated_model.pkl")
    cross_path = os.path.join(organ_dir, "cross_organ_model.pkl")

    if not os.path.exists(iso_path) or not os.path.exists(cross_path):
        raise FileNotFoundError(
            f"Pre-trained models not found in {organ_dir}!\n"
            f"Please run 'python pipeline/{organ_key.lower()}.py' first."
        )

    with open(iso_path, "rb") as f:
        iso_model = pickle.load(f)

    with open(cross_path, "rb") as f:
        cross_model = pickle.load(f)

    return iso_model, cross_model


def preprocess_all_patient_inputs(raw_dict: dict) -> pd.DataFrame:
    """Computes derived variables, log transforms, and eGFR calculation."""
    p = raw_dict.copy()

    for k, v in list(p.items()):
        try:
            p[k] = float(v)
        except (ValueError, TypeError):
            pass

    # Safe defaults
    p.setdefault("Age", 55.0)
    p.setdefault("Sex_1M_2F", 1.0)
    p.setdefault("Systolic_BP_mmHg", 125.0)
    p.setdefault("Diastolic_BP_mmHg", 78.0)
    p.setdefault("hs_CRP", 2.0)
    p.setdefault("Serum_Creatinine_mg_dL", 1.0)
    p.setdefault("BUN_mg_dL", 15.0)
    p.setdefault("ACR", 20.0)
    p.setdefault("Urine_Albumin", 25.0)
    p.setdefault("HbA1c_pct", 5.6)
    p.setdefault("Fasting_Glucose_mg_dL", 100.0)
    p.setdefault("Insulin_Use", 0.0)
    p.setdefault("ALT_U_L", 25.0)
    p.setdefault("AST_U_L", 24.0)
    p.setdefault("Total_Bilirubin_mg_dL", 0.7)
    p.setdefault("Platelets_1000_uL", 250.0)
    p.setdefault("Albumin_g_dL", 4.2)
    p.setdefault("Liver_Fat_CAP_dB_m", 230.0)
    p.setdefault("BMI", 27.0)
    p.setdefault("Waist_Circumference_cm", 90.0)

    # Derived Log Transforms
    p["log_hs_CRP"] = float(np.log1p(max(0.0, float(p["hs_CRP"]))))
    p["log_Serum_Creatinine"] = float(np.log1p(max(0.0, float(p["Serum_Creatinine_mg_dL"]))))
    p["log_Albumin_Creatinine_Ratio_ACR"] = float(np.log1p(max(0.0, float(p["ACR"]))))
    p["log_Urine_Albumin"] = float(np.log1p(max(0.0, float(p["Urine_Albumin"]))))
    p["log_ALT"] = float(np.log1p(max(0.0, float(p["ALT_U_L"]))))
    p["log_AST"] = float(np.log1p(max(0.0, float(p["AST_U_L"]))))
    p["log_Total_Bilirubin"] = float(np.log1p(max(0.0, float(p["Total_Bilirubin_mg_dL"]))))

    # Compute eGFR (CKD-EPI 2021)
    if "eGFR" not in p or p["eGFR"] is None:
        scr = float(p["Serum_Creatinine_mg_dL"])
        age = float(p["Age"])
        sex = float(p["Sex_1M_2F"])
        p["eGFR"] = float(heart.compute_egfr_ckd_epi_2021(scr, age, sex))

    return pd.DataFrame([p])


def evaluate_organ_prediction(organ_key: str, df_patient: pd.DataFrame) -> dict:
    """
    Evaluates:
      a) Layer 1: Target organ parameters only
      b) Layer 2: Target organ + all external organs combined
      c) Pairwise Crosstalk: (Target, Other Organ) for each external organ
    """
    spec = ORGAN_REGISTRY[organ_key]
    iso_m, cross_m = load_organ_models(organ_key)
    device = torch.device("cpu")

    # ─────────────────────────────────────────────────────────────────
    # a) Layer 1 Risk (Target Organ Only)
    # ─────────────────────────────────────────────────────────────────
    tgt_cols = iso_m["features"]
    X_tgt_scaled = iso_m["scaler"].transform(iso_m["imputer"].transform(df_patient[tgt_cols]))
    l1_prob = float(iso_m["model"].predict_proba(X_tgt_scaled)[0, 1])

    # ─────────────────────────────────────────────────────────────────
    # b) Layer 2 Risk (All Organs Combined)
    # ─────────────────────────────────────────────────────────────────
    dep_preprocessors = cross_m.get("dep_models", {})
    l2_net = cross_m["best_model"]
    l2_net.eval()

    x_tgt_t = torch.tensor(X_tgt_scaled, dtype=torch.float32, device=device)

    x_deps_full = {}
    for dep_org, cols in spec["dep_cols"].items():
        if dep_org in dep_preprocessors:
            pp = dep_preprocessors[dep_org]
            dep_scaled = pp["scaler"].transform(pp["imputer"].transform(df_patient[cols]))
            x_deps_full[dep_org] = torch.tensor(dep_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = l2_net(x_tgt_t, x_deps_full)
        logit = out[0] if isinstance(out, tuple) else out
        l2_prob = float(torch.sigmoid(logit).item())

    # ─────────────────────────────────────────────────────────────────
    # c) Pairwise Crosstalk Risk Scores: (Target, External Organ)
    # ─────────────────────────────────────────────────────────────────
    pairwise_scores = {}
    for dep_org in spec["dep_cols"]:
        x_pair = {}
        for other_dep in spec["dep_cols"]:
            if other_dep == dep_org:
                x_pair[other_dep] = x_deps_full[other_dep].clone()
            else:
                # Neutral baseline (0 in standardized z-score space)
                x_pair[other_dep] = torch.zeros_like(x_deps_full[other_dep])

        with torch.no_grad():
            res_p = l2_net(x_tgt_t, x_pair)
            l_p = res_p[0] if isinstance(res_p, tuple) else res_p
            p_prob = float(torch.sigmoid(l_p).item())
            p_delta = p_prob - l1_prob
            p_rel_delta = (p_delta / l1_prob * 100.0) if l1_prob > 1e-6 else 0.0
            p_error_pct = (abs(p_delta) / l1_prob * 100.0) if l1_prob > 1e-6 else 0.0
            pairwise_scores[dep_org] = {
                "pair_name": f"({spec['name']}, {dep_org})",
                "risk_prob": p_prob,
                "delta": p_delta,
                "rel_delta_pct": p_rel_delta,
                "error_pct": p_error_pct
            }

    abs_delta = l2_prob - l1_prob
    rel_delta_pct = (abs_delta / l1_prob * 100.0) if l1_prob > 1e-6 else 0.0
    error_pct = (abs(abs_delta) / l1_prob * 100.0) if l1_prob > 1e-6 else 0.0
    abs_error = abs(abs_delta)

    return {
        "organ_key": organ_key,
        "name": spec["name"],
        "endpoint": spec["endpoint"],
        "description": spec["description"],
        "layer1_model": iso_m["best_model_name"],
        "layer1_risk": l1_prob,
        "layer1_tier": compute_risk_tier(l1_prob),
        "layer1_features": tgt_cols,
        "layer2_model": cross_m["best_layer2_model_name"],
        "layer2_risk": l2_prob,
        "layer2_tier": compute_risk_tier(l2_prob),
        "delta": abs_delta,
        "rel_delta_pct": rel_delta_pct,
        "error_pct": error_pct,
        "abs_error": abs_error,
        "pairwise_scores": pairwise_scores
    }


def display_organ_report(res: dict):
    """Prints the requested (a), (b), and (c) structured output for a specific organ with delta & error percentages."""
    name = res["name"]
    endpoint = res["endpoint"]
    l1 = res["layer1_risk"]
    l2 = res["layer2_risk"]
    delta = res["delta"]
    rel_delta = res["rel_delta_pct"]
    err_pct = res["error_pct"]

    print("\n" + "=" * 78)
    print(f"   PREDICTION RESULTS FOR {name.upper()} ({endpoint.upper()})")
    print("=" * 78)

    # a) Layer 1 Risk Score
    print(f"\n a) RISK SCORE FOR LAYER 1 (Only {name} Parameters Used):")
    print(f"    • Model Architecture   : {res['layer1_model']}")
    print(f"    • Biomarkers Evaluated : {', '.join(res['layer1_features'])}")
    print(f"    • Layer 1 Risk Score   : {render_progress_bar(l1)}")
    print(f"    • Risk Classification  : {res['layer1_tier']}")

    # b) Layer 2 Risk Score
    print(f"\n b) RISK SCORE FOR LAYER 2 ({name} + All External Organs Combined):")
    print(f"    • Model Architecture   : {res['layer2_model']}")
    print(f"    • Integrated Organs    : {name} + " + ", ".join(res["pairwise_scores"].keys()))
    print(f"    • Layer 2 Risk Score   : {render_progress_bar(l2)}")
    print(f"    • Risk Classification  : {res['layer2_tier']}")

    # Dedicated Delta & Error Percentage Box
    print(f"\n    ────────────────────────────────────────────────────────────────────────")
    print(f"    ► DELTA & MODEL DISCREPANCY (ERROR / DIFFERENCE PERCENTAGE):")
    print(f"       • Absolute Difference (Δ)     : {delta*100:+6.2f}%  (Layer 2 − Layer 1)")
    print(f"       • Absolute Error Divergence   : {res['abs_error']:.4f}  (on 0.0–1.0 probability scale)")
    print(f"    ────────────────────────────────────────────────────────────────────────")

    # c) Pairwise Cross-Organ Crosstalk Scores
    print(f"\n c) PAIRWISE CROSS-ORGAN CROSSTALK RISK SCORES FOR {endpoint.upper()}:")
    print(f"    {'Organ Pair Interaction':<24} | {'Pairwise Risk':<12} | {'Absolute Δ':<12} | {'Relative Δ%':<14} | {'Discrepancy %'}")
    print("    " + "-" * 76)
    for dep_org, pair_info in res["pairwise_scores"].items():
        pair_prob = pair_info["risk_prob"]
        pair_delta = pair_info["delta"]
        pair_rel_delta = pair_info["rel_delta_pct"]
        pair_err = pair_info["error_pct"]
        print(f"    {pair_info['pair_name']:<24} | {render_progress_bar(pair_prob, width=12)} | {pair_delta*100:+6.2f}%     | {pair_rel_delta:+7.2f}%      | {pair_err:6.2f}%")

    print("\n" + "=" * 78)


def generate_patient_crosstalk_heatmap(all_reports: dict, save_path: str = None) -> str:
    """
    Generates a 5x5 publication-quality Patient-Specific Crosstalk Heatmap:
      - Diagonal: Layer 1 baseline risk for each organ
      - Off-diagonal (i, j): Pairwise modulated risk (%) and delta percentage shift (Δ%)
        when external organ j interacts with target organ i.
    """
    if save_path is None:
        save_path = os.path.join(REPO_ROOT, "pipeline", "outputs", "figures", "patient_pairwise_crosstalk_heatmap.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    org_keys = ["heart", "kidney", "pancreas", "liver", "adipose"]
    org_names = ["Heart", "Kidney", "Pancreas", "Liver", "Adipose"]
    n = len(org_keys)

    delta_matrix = np.zeros((n, n), dtype=float)
    annot_matrix = np.empty((n, n), dtype=object)

    for i, t_key in enumerate(org_keys):
        t_rep = all_reports.get(t_key)
        if not t_rep:
            continue
        l1 = t_rep["layer1_risk"]
        for j, dep_name in enumerate(org_names):
            if i == j:
                delta_matrix[i, j] = 0.0
                annot_matrix[i, j] = f"{l1*100:.1f}%\n(Baseline)"
            else:
                pair_info = t_rep.get("pairwise_scores", {}).get(dep_name)
                if pair_info:
                    p_risk = pair_info["risk_prob"]
                    d = pair_info["delta"]
                    delta_matrix[i, j] = d * 100.0
                    annot_matrix[i, j] = f"{p_risk*100:.1f}%\n({d*100:+.1f}%)"
                else:
                    delta_matrix[i, j] = 0.0
                    annot_matrix[i, j] = "—"

    fig, ax = plt.subplots(figsize=(10.5, 8.8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    heatmap = sns.heatmap(
        delta_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        center=0.0,
        vmin=-25.0,
        vmax=+25.0,
        ax=ax,
        cbar_kws={"label": "Pairwise Risk Shift (Δ% = Layer 2 Pair − Layer 1 Baseline)", "shrink": 0.85},
        linewidths=1.8,
        linecolor="#1e222d",
        annot_kws={"fontsize": 10.5, "fontweight": "bold", "color": "white"}
    )

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels([f"Ext. {name}" for name in org_names], fontsize=11, color="#e0e0e0")
    ax.set_yticklabels([f"Target {name}" for name in org_names], fontsize=11, color="#e0e0e0", rotation=0)

    ax.set_title("Patient-Specific 5-Organ Crosstalk Risk Shift Heatmap\n(Diagonal = Layer 1 Baseline | Off-Diagonal = Pairwise Modulated Risk & Δ%)",
                 fontsize=13.5, fontweight="bold", pad=16, color="#ffffff")

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#e0e0e0")
    cbar.ax.yaxis.label.set_size(11)
    cbar.ax.tick_params(colors="#e0e0e0", labelsize=10)

    plt.figtext(0.5, 0.02,
                "▲ Warm/Red = External organ amplifies target disease risk | ▼ Cool/Blue = External organ attenuates/buffers risk",
                ha="center", fontsize=10, color="#9ca3af", fontstyle="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    return save_path


def display_all_organs_table(all_reports: dict):
    """Prints a consolidated high-level table across all 5 organ systems with delta and error metrics."""
    print("\n" + "=" * 105)
    print("              CONSOLIDATED 5-ORGAN RISK, DELTA & DISCREPANCY / ERROR PERCENTAGE SUMMARY")
    print("=" * 105)
    print(f"{'Organ Target':<14} | {'Disease Endpoint':<24} | {'Layer 1':<8} | {'Layer 2':<8} | {'Absolute Δ':<11} | {'Relative Δ%':<12} | {'Discrepancy %':<14} | {'Risk Effect'}")
    print("-" * 105)
    for org_key, report in all_reports.items():
        l1_str = f"{report['layer1_risk']*100:5.1f}%"
        l2_str = f"{report['layer2_risk']*100:5.1f}%"
        d_str = f"{report['delta']*100:+5.1f}%"
        rel_str = f"{report['rel_delta_pct']:+5.1f}%"
        err_str = f"{report['error_pct']:5.1f}%"
        if report['delta'] > 0.005:
            effect = "▲ Amplified"
        elif report['delta'] < -0.005:
            effect = "▼ Attenuated"
        else:
            effect = "= Concordant"
        print(f"{report['name']:<14} | {report['endpoint']:<24} | {l1_str:<8} | {l2_str:<8} | {d_str:<11} | {rel_str:<12} | {err_str:<14} | {effect}")
    print("=" * 105)

    heatmap_path = generate_patient_crosstalk_heatmap(all_reports)
    print(f" ✓ Generated Patient-Specific Crosstalk Heatmap: {heatmap_path}\n")


def prompt_all_patient_inputs() -> dict:
    """Interactively gathers all clinical biomarker inputs from the user."""
    print("\n" + "=" * 76)
    print(" STEP 1: ENTER PATIENT CLINICAL BIOMARKERS")
    print(" Press [ENTER] to accept the suggested default value in brackets.")
    print("=" * 76)

    def ask(prompt_text, default_val, val_type=float):
        user_val = input(f" {prompt_text} [{default_val}]: ").strip()
        if not user_val:
            return val_type(default_val)
        try:
            return val_type(user_val)
        except ValueError:
            print(f"   Invalid input, using default: {default_val}")
            return val_type(default_val)

    print("\n--- [Patient Demographics & Vitals] ---")
    age = ask("Enter Patient Age (years)", 58.0)
    sex = ask("Enter Patient Sex (1 = Male, 2 = Female)", 1.0)
    sys_bp = ask("Enter Systolic Blood Pressure (mmHg)", 145.0)
    dia_bp = ask("Enter Diastolic Blood Pressure (mmHg)", 88.0)
    hs_crp = ask("Enter hs-CRP (mg/L, high-sensitivity C-reactive protein)", 3.2)

    print("\n--- [Kidney Biomarkers (Cardiorenal Axis)] ---")
    creatinine = ask("Enter Serum Creatinine (mg/dL)", 1.2)
    bun = ask("Enter Blood Urea Nitrogen / BUN (mg/dL)", 18.0)
    acr = ask("Enter Urine Albumin-to-Creatinine Ratio / ACR (mg/g)", 35.0)
    u_alb = ask("Enter Urine Albumin (mg/L)", 42.0)

    print("\n--- [Pancreas Biomarkers (Glycemic Axis)] ---")
    hba1c = ask("Enter HbA1c (%)", 6.8)
    glucose = ask("Enter Fasting Blood Glucose (mg/dL)", 115.0)
    insulin = ask("Enter Insulin Therapy Use (1 = Yes, 0 = No)", 0.0)

    print("\n--- [Liver Biomarkers (Hepatic Axis)] ---")
    alt = ask("Enter ALT / SGPT (U/L)", 38.0)
    ast = ask("Enter AST / SGOT (U/L)", 32.0)
    cap = ask("Enter Liver Fat Steatosis CAP (dB/m)", 275.0)
    bili = ask("Enter Total Bilirubin (mg/dL)", 0.8)
    plat = ask("Enter Platelet Count (1000/uL)", 240.0)
    alb = ask("Enter Serum Albumin (g/dL)", 4.1)

    print("\n--- [Adipose Biomarkers (Metabolic Adiposity)] ---")
    bmi = ask("Enter Body Mass Index / BMI (kg/m²)", 29.5)
    waist = ask("Enter Waist Circumference (cm)", 98.0)

    return {
        "Age": age,
        "Sex_1M_2F": sex,
        "Systolic_BP_mmHg": sys_bp,
        "Diastolic_BP_mmHg": dia_bp,
        "hs_CRP": hs_crp,
        "Serum_Creatinine_mg_dL": creatinine,
        "BUN_mg_dL": bun,
        "ACR": acr,
        "Urine_Albumin": u_alb,
        "HbA1c_pct": hba1c,
        "Fasting_Glucose_mg_dL": glucose,
        "Insulin_Use": insulin,
        "ALT_U_L": alt,
        "AST_U_L": ast,
        "Liver_Fat_CAP_dB_m": cap,
        "Total_Bilirubin_mg_dL": bili,
        "Platelets_1000_uL": plat,
        "Albumin_g_dL": alb,
        "BMI": bmi,
        "Waist_Circumference_cm": waist
    }


def organ_selection_loop(df_patient: pd.DataFrame):
    """
    Prompts the user to select an organ and displays:
      a) Layer 1 risk score
      b) Layer 2 risk score
      c) Pairwise crosstalk risk scores
    Loops so the user can query multiple organs on the same patient.
    """
    code_to_key = {
        "H": "heart",
        "K": "kidney",
        "L": "liver",
        "P": "pancreas",
        "A": "adipose",
    }

    while True:
        print("\n" + "=" * 76)
        print(" STEP 2: SELECT SPECIFIC TARGET ORGAN FOR RISK PREDICTION")
        print("=" * 76)
        print("  [H] Heart     — Hypertension Risk")
        print("  [K] Kidney    — Albuminuria / Chronic Kidney Disease Risk")
        print("  [L] Liver     — MASLD / Hepatic Steatosis Risk")
        print("  [P] Pancreas  — Adenocarcinoma / Pancreatic Disease Risk")
        print("  [A] Adipose   — Visceral Adiposity / Obesity Risk")
        print("  [ALL] Run All 5 Organs Simultaneously")
        print("  [Q] Exit")
        print("-" * 76)

        choice = input(" Enter your choice (H, K, L, P, A, ALL, or Q to quit): ").strip().upper()

        if choice == "Q" or choice == "QUIT":
            print("\n Exiting Patient Risk Predictor. Thank you!\n")
            break

        if choice in code_to_key:
            organ_key = code_to_key[choice]
            report = evaluate_organ_prediction(organ_key, df_patient)
            display_organ_report(report)

        elif choice == "ALL":
            all_reports = {}
            for org_key in ORGAN_REGISTRY:
                all_reports[org_key] = evaluate_organ_prediction(org_key, df_patient)
            display_all_organs_table(all_reports)

        elif choice.lower() in ORGAN_REGISTRY:
            organ_key = choice.lower()
            report = evaluate_organ_prediction(organ_key, df_patient)
            display_organ_report(report)

        else:
            print(f" Invalid choice '{choice}'. Please enter H, K, L, P, A, ALL, or Q.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Organ Patient Risk Predictor (Layer 1, Layer 2 & Pairwise)")
    parser.add_argument("--demo", action="store_true", help="Run with standard sample patient profile")
    parser.add_argument("--organ", type=str, default=None, choices=["H", "K", "L", "P", "A", "ALL", "heart", "kidney", "liver", "pancreas", "adipose"], help="Target organ choice")
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--sex", type=float, default=None)
    parser.add_argument("--systolic_bp", type=float, default=None)
    parser.add_argument("--diastolic_bp", type=float, default=None)
    parser.add_argument("--glucose", type=float, default=None)
    parser.add_argument("--hba1c", type=float, default=None)
    parser.add_argument("--creatinine", type=float, default=None)
    parser.add_argument("--bmi", type=float, default=None)
    parser.add_argument("--waist", type=float, default=None)

    args = parser.parse_args()

    if args.demo:
        print("\n [DEMO MODE] Using Patient Profile (Age: 58, Systolic BP: 145, Fasting Glucose: 115)...")
        patient_inputs = {
            "Age": 58.0, "Sex_1M_2F": 1.0, "Systolic_BP_mmHg": 145.0, "Diastolic_BP_mmHg": 88.0,
            "hs_CRP": 3.2, "Serum_Creatinine_mg_dL": 1.2, "BUN_mg_dL": 18.0, "ACR": 35.0,
            "Urine_Albumin": 42.0, "HbA1c_pct": 6.8, "Fasting_Glucose_mg_dL": 115.0, "Insulin_Use": 0.0,
            "ALT_U_L": 38.0, "AST_U_L": 32.0, "Liver_Fat_CAP_dB_m": 275.0,
            "Total_Bilirubin_mg_dL": 0.8, "Platelets_1000_uL": 240.0, "Albumin_g_dL": 4.1,
            "BMI": 29.5, "Waist_Circumference_cm": 98.0
        }
    elif args.age is not None:
        patient_inputs = {
            "Age": args.age,
            "Sex_1M_2F": args.sex if args.sex is not None else 1.0,
            "Systolic_BP_mmHg": args.systolic_bp if args.systolic_bp is not None else 145.0,
            "Diastolic_BP_mmHg": args.diastolic_bp if args.diastolic_bp is not None else 88.0,
            "hs_CRP": 3.2,
            "Serum_Creatinine_mg_dL": args.creatinine if args.creatinine is not None else 1.2,
            "BUN_mg_dL": 18.0,
            "ACR": 35.0,
            "Urine_Albumin": 42.0,
            "HbA1c_pct": args.hba1c if args.hba1c is not None else 6.8,
            "Fasting_Glucose_mg_dL": args.glucose if args.glucose is not None else 115.0,
            "Insulin_Use": 0.0,
            "ALT_U_L": 38.0,
            "AST_U_L": 32.0,
            "Liver_Fat_CAP_dB_m": 275.0,
            "Total_Bilirubin_mg_dL": 0.8,
            "Platelets_1000_uL": 240.0,
            "Albumin_g_dL": 4.1,
            "BMI": args.bmi if args.bmi is not None else 29.5,
            "Waist_Circumference_cm": args.waist if args.waist is not None else 98.0
        }
    else:
        patient_inputs = prompt_all_patient_inputs()

    df_patient = preprocess_all_patient_inputs(patient_inputs)

    if args.organ:
        code_to_key = {"H": "heart", "K": "kidney", "L": "liver", "P": "pancreas", "A": "adipose"}
        choice = args.organ.upper()
        if choice in code_to_key:
            report = evaluate_organ_prediction(code_to_key[choice], df_patient)
            display_organ_report(report)
        elif choice == "ALL":
            all_reports = {k: evaluate_organ_prediction(k, df_patient) for k in ORGAN_REGISTRY}
            display_all_organs_table(all_reports)
        elif args.organ.lower() in ORGAN_REGISTRY:
            report = evaluate_organ_prediction(args.organ.lower(), df_patient)
            display_organ_report(report)
    else:
        organ_selection_loop(df_patient)


if __name__ == "__main__":
    main()
