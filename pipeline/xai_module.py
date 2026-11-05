"""
pipeline/xai_module.py
======================
Shared Explainable AI (XAI) module for post-Layer 2 analysis.
Called by each organ pipeline after Layer 2 training completes.

Produces:
  1. SHAP analysis on Layer 1 best model (Random Forest / tree-based)
  2. Gradient-based feature attribution on Layer 2 DL model
  3. Combined feature importance ranking across all features
  4. Per-organ contribution waterfall chart
  5. CSV export of all importance values
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

# Organ color palette (consistent across all pipelines)
ORGAN_COLOR_MAP = {
    "Heart (Target)": "#e63946",   "Heart": "#e63946",
    "Kidney (Target)": "#457b9d",  "Kidney": "#457b9d",
    "Liver (Target)": "#2a9d8f",   "Liver": "#2a9d8f",
    "Pancreas (Target)": "#f4a261","Pancreas": "#f4a261",
    "Adipose (Target)": "#e76f51", "Adipose": "#e76f51",
}


def run_explainable_ai(organ_name, target_label_col,
                       train_df, test_df,
                       isolated_result, enhanced_result,
                       target_cols, dep_cols,
                       out_dir, fig_dir, dpi=150,
                       random_seed=42):
    """
    Universal XAI function for any organ pipeline.

    Parameters
    ----------
    organ_name : str
        e.g., "Heart", "Kidney", "Liver", "Pancreas", "Adipose"
    target_label_col : str
        e.g., "Hypertension_Label", "Albuminuria_Label", etc.
    train_df, test_df : pd.DataFrame
    isolated_result : dict  (Layer 1 output)
    enhanced_result : dict  (Layer 2 output)
    target_cols : list[str]
    dep_cols : dict[str, list[str]]
    out_dir, fig_dir : str
    dpi : int
    random_seed : int
    """
    print("\n" + "="*70)
    print(f" [XAI] EXPLAINABLE AI — POST-LAYER 2 ANALYSIS ({organ_name.upper()})")
    print("="*70)

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    y_test = test_df[target_label_col].values
    y_train = train_df[target_label_col].values

    target_source = f"{organ_name} (Target)"

    # ── Prepare feature matrices ─────────────────────────────────────────
    imputer = isolated_result["imputer"]
    scaler = isolated_result["scaler"]
    X_test_target = scaler.transform(imputer.transform(test_df[target_cols]))
    X_train_target = scaler.transform(imputer.transform(train_df[target_cols]))

    dep_preprocessors = enhanced_result.get("dep_models", {})

    # Combined feature matrix
    X_test_combined_parts = [X_test_target]
    X_train_combined_parts = [X_train_target]
    for organ, cols in dep_cols.items():
        if len(cols) == 0 or organ not in dep_preprocessors:
            continue
        pp = dep_preprocessors[organ]
        X_test_combined_parts.append(pp["scaler"].transform(pp["imputer"].transform(test_df[cols])))
        X_train_combined_parts.append(pp["scaler"].transform(pp["imputer"].transform(train_df[cols])))

    importance_records = []

    # ═══════════════════════════════════════════════════════════════════════
    # 1. SHAP ANALYSIS ON LAYER 1 BEST MODEL
    # ═══════════════════════════════════════════════════════════════════════
    print("\n [XAI Step 1/4] SHAP Analysis on Layer 1 Model...")
    best_model = isolated_result["best_model"]
    best_name = isolated_result["best_name"]

    shap_values_arr = None
    if HAS_SHAP:
        try:
            try:
                explainer = shap.TreeExplainer(best_model)
                shap_values_obj = explainer(X_test_target)
            except Exception:
                # Fallback to general Explainer for non-tree models
                explainer = shap.Explainer(best_model, X_train_target[:min(100, len(X_train_target))])
                shap_values_obj = explainer(X_test_target)

            if hasattr(shap_values_obj, "values"):
                vals = shap_values_obj.values
                if len(vals.shape) == 3 and vals.shape[2] == 2:
                    shap_values_arr = vals[:, :, 1]
                elif len(vals.shape) == 3 and vals.shape[2] == 1:
                    shap_values_arr = vals[:, :, 0]
                else:
                    shap_values_arr = vals
            else:
                shap_values_arr = np.asarray(shap_values_obj)

            mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=0)
            for i, col in enumerate(target_cols):
                importance_records.append({
                    "Feature": col,
                    "Organ_Source": target_source,
                    "Method": "SHAP (Layer 1)",
                    "Importance": round(float(mean_abs_shap[i]), 6)
                })

            base_val = getattr(shap_values_obj, "base_values", 0.0)
            if hasattr(base_val, "shape") and len(base_val.shape) > 1 and base_val.shape[1] > 1:
                base_val = base_val[:, 1]

            # SHAP Beeswarm Plot
            plt.figure(figsize=(9, max(4, len(target_cols) * 0.5 + 1)))
            shap_display = shap.Explanation(
                values=shap_values_arr,
                base_values=base_val,
                data=X_test_target,
                feature_names=target_cols
            )
            shap.plots.beeswarm(shap_display, show=False, max_display=len(target_cols))
            plt.title(f"{organ_name} — SHAP Beeswarm (Layer 1: {best_name})",
                      fontsize=13, fontweight="bold", pad=14)
            plt.tight_layout()
            shap_bee_path = os.path.join(fig_dir, "xai_shap_summary.png")
            plt.savefig(shap_bee_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            print(f"   ✓ SHAP beeswarm plot saved: {shap_bee_path}")

            # SHAP Bar Plot
            plt.figure(figsize=(8, max(3.5, len(target_cols) * 0.45 + 1)))
            shap.plots.bar(shap_display, show=False, max_display=len(target_cols))
            plt.title(f"{organ_name} — SHAP Feature Importance (Layer 1: {best_name})",
                      fontsize=13, fontweight="bold", pad=14)
            plt.tight_layout()
            shap_bar_path = os.path.join(fig_dir, "xai_shap_bar.png")
            plt.savefig(shap_bar_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            print(f"   ✓ SHAP bar plot saved: {shap_bar_path}")

        except Exception as e:
            print(f"   ⚠ SHAP analysis failed ({e}), falling back to permutation importance.")
            shap_values_arr = None

    if shap_values_arr is None:
        # Fallback: Permutation Importance
        print("   → Using permutation importance (SHAP fallback)...")
        from sklearn.inspection import permutation_importance
        perm_result = permutation_importance(
            best_model, X_test_target, y_test,
            n_repeats=20, random_state=random_seed, scoring="roc_auc"
        )
        perm_imp = perm_result.importances_mean
        for i, col in enumerate(target_cols):
            importance_records.append({
                "Feature": col,
                "Organ_Source": target_source,
                "Method": "Permutation Importance (Layer 1)",
                "Importance": round(float(perm_imp[i]), 6)
            })

        sorted_idx = np.argsort(perm_imp)[::-1]
        fig, ax = plt.subplots(figsize=(8, max(3.5, len(target_cols) * 0.45 + 1)))
        ax.barh([target_cols[i] for i in sorted_idx], perm_imp[sorted_idx],
                color="#2b5c8f", edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Mean Permutation Importance (AUROC drop)", fontsize=11, fontweight="bold")
        ax.set_title(f"{organ_name} — Permutation Importance (Layer 1: {best_name})",
                     fontsize=13, fontweight="bold", pad=14)
        ax.invert_yaxis()
        plt.tight_layout()
        perm_path = os.path.join(fig_dir, "xai_shap_summary.png")
        plt.savefig(perm_path, dpi=dpi, bbox_inches="tight")
        perm_bar_path = os.path.join(fig_dir, "xai_shap_bar.png")
        plt.savefig(perm_bar_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Permutation importance plot saved: {perm_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. GRADIENT-BASED FEATURE ATTRIBUTION ON LAYER 2 DL MODEL
    # ═══════════════════════════════════════════════════════════════════════
    print("\n [XAI Step 2/4] Gradient-Based Attribution on Layer 2 DL Model...")
    l2_model = enhanced_result["best_model"]
    l2_name = enhanced_result["best_name"]

    organ_color_map = ORGAN_COLOR_MAP.copy()

    try:
        device = torch.device("cpu")
        l2_model.eval()

        x_tgt = torch.tensor(X_test_target, dtype=torch.float32, device=device, requires_grad=True)
        x_deps_dict = {}
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
        forward_res = l2_model(x_tgt, x_deps_dict)
        logits = forward_res[0] if isinstance(forward_res, tuple) else forward_res
        probs = torch.sigmoid(logits)

        # Compute gradients
        grad_outputs = torch.ones_like(probs)
        all_inputs = [x_tgt] + [dep_grad_tensors[o] for o in dep_grad_tensors]
        grads = torch.autograd.grad(probs, all_inputs, grad_outputs=grad_outputs,
                                     create_graph=False, allow_unused=True)

        # Target organ gradient importance
        if grads[0] is not None:
            tgt_grad = grads[0].detach().cpu().numpy()
            tgt_attribution = np.mean(np.abs(tgt_grad * X_test_target), axis=0)
        else:
            tgt_attribution = np.zeros(len(target_cols))

        for i, col in enumerate(target_cols):
            importance_records.append({
                "Feature": col,
                "Organ_Source": target_source,
                "Method": "Gradient Attribution (Layer 2)",
                "Importance": round(float(tgt_attribution[i]), 6)
            })

        # Dependency organ gradient importance
        grad_idx = 1
        for organ in dep_grad_tensors:
            cols = dep_cols[organ]
            pp = dep_preprocessors[organ]
            X_dep = pp["scaler"].transform(pp["imputer"].transform(test_df[cols]))
            if grads[grad_idx] is not None:
                dep_grad = grads[grad_idx].detach().cpu().numpy()
                dep_attribution = np.mean(np.abs(dep_grad * X_dep), axis=0)
            else:
                dep_attribution = np.zeros(len(cols))
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

        bar_colors = [organ_color_map.get(s, "#999999") for s in l2_df["Organ_Source"]]

        fig, ax = plt.subplots(figsize=(9, max(5, len(l2_df) * 0.4 + 1.5)))
        ax.barh(l2_df["Feature"], l2_df["Importance"], color=bar_colors,
                edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Mean |Gradient × Input| Attribution", fontsize=11, fontweight="bold")
        ax.set_title(f"{organ_name} — Layer 2 Feature Attribution ({l2_name})",
                     fontsize=13, fontweight="bold", pad=14)

        legend_elements = [Patch(facecolor=c, edgecolor="black", label=n)
                           for n, c in organ_color_map.items()
                           if n in l2_df["Organ_Source"].values]
        if legend_elements:
            ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        l2_attr_path = os.path.join(fig_dir, "xai_layer2_feature_attribution.png")
        plt.savefig(l2_attr_path, dpi=dpi, bbox_inches="tight")
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
        ax.set_xlabel("Average Normalized Importance (across XAI methods)",
                      fontsize=11, fontweight="bold")
        ax.set_title(f"{organ_name} — Combined Feature Importance (Layer 1 + Layer 2)",
                     fontsize=13, fontweight="bold", pad=14)

        legend_elements = [Patch(facecolor=c, edgecolor="black", label=n)
                           for n, c in organ_color_map.items()
                           if n in agg_df["Organ_Source"].values]
        if legend_elements:
            ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        combined_path = os.path.join(fig_dir, "xai_combined_feature_importance.png")
        plt.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Combined feature importance plot saved: {combined_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PER-ORGAN CONTRIBUTION WATERFALL
    # ═══════════════════════════════════════════════════════════════════════
    print("\n [XAI Step 4/4] Per-Organ Contribution Waterfall...")
    if not imp_df.empty:
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
        ax.set_title(f"{organ_name} — Per-Organ Feature Contribution (XAI)",
                     fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()
        waterfall_path = os.path.join(fig_dir, "xai_organ_contribution_waterfall.png")
        plt.savefig(waterfall_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Per-organ contribution waterfall saved: {waterfall_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    if importance_records:
        xai_csv = pd.DataFrame(importance_records)
        xai_csv_path = os.path.join(out_dir, "xai_feature_importance.csv")
        xai_csv.to_csv(xai_csv_path, index=False)
        print(f"   ✓ XAI feature importance CSV saved: {xai_csv_path}")

    # ── Console Summary ──────────────────────────────────────────────────
    print("\n   " + "="*60)
    print(f"   EXPLAINABLE AI SUMMARY — {organ_name.upper()}")
    print("   " + "-"*60)
    if not imp_df.empty:
        for method in imp_df["Method"].unique():
            print(f"\n   [{method}]:")
            m_df = imp_df[imp_df["Method"] == method].sort_values("Importance", ascending=False)
            for _, row in m_df.iterrows():
                print(f"     {row['Feature']:35s} ({row['Organ_Source']:20s}): {row['Importance']:.6f}")
    print("   " + "="*60)

    return importance_records


def main():
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="Multi-Organ Explainable AI (XAI) Runner")
    parser.add_argument(
        "--organ", type=str, default="all",
        choices=["all", "heart", "kidney", "liver", "pancreas", "adipose"],
        help="Organ to run XAI for (default: all)"
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_dir = os.path.join(repo_root, "pipeline")

    organs_map = {
        "heart": ("heart.py", "Heart Organ Pipeline (Hypertension Risk)"),
        "kidney": ("kidney.py", "Kidney Organ Pipeline (Albuminuria / CKD Risk)"),
        "liver": ("liver.py", "Liver Organ Pipeline (MASLD / Steatosis Risk)"),
        "pancreas": ("pancreas.py", "Pancreas Organ Pipeline (Adenocarcinoma Risk)"),
        "adipose": ("adipose.py", "Adipose Organ Pipeline (Visceral Adiposity Risk)"),
    }

    if args.organ == "all":
        targets = list(organs_map.keys())
    else:
        targets = [args.organ.lower()]

    print("=" * 75)
    print(" MULTI-ORGAN EXPLAINABLE AI (XAI) — POST-LAYER 2 MASTER RUNNER")
    print(f" Targets: {', '.join([t.capitalize() for t in targets])}")
    print(f" Python Executable: {sys.executable}")
    print("=" * 75)

    for organ in targets:
        script, desc = organs_map[organ]
        script_path = os.path.join(pipeline_dir, script)
        print("\n" + "=" * 75)
        print(f" [XAI EXECUTION] {organ.upper()} — {desc}")
        print("=" * 75)
        res = subprocess.run([sys.executable, script_path], cwd=repo_root)
        if res.returncode != 0:
            print(f"\n [ERROR] XAI execution failed for {organ} with code {res.returncode}!")
            sys.exit(res.returncode)

    print("\n" + "=" * 75)
    print(" ALL REQUESTED EXPLAINABLE AI ANALYSES COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    print("\n Generated XAI Outputs Summary:")
    for organ in targets:
        organ_out = os.path.join(pipeline_dir, "outputs", organ)
        fig_dir = os.path.join(organ_out, "figures")
        csv_path = os.path.join(organ_out, "xai_feature_importance.csv")
        print(f"\n   * {organ.capitalize()}:")
        if os.path.exists(csv_path):
            print(f"     - CSV: {csv_path}")
        if os.path.exists(fig_dir):
            xai_figs = [f for f in os.listdir(fig_dir) if f.startswith("xai_")]
            for fig in sorted(xai_figs):
                print(f"     - Fig: {os.path.join(fig_dir, fig)}")
    print("\n" + "=" * 75 + "\n")


if __name__ == "__main__":
    main()
