"""
Main entry point for Heart Disease Risk Prediction using Calibrated Deep Learning.
Publication: Engineered Science (Q1 Journal, ES-2026-0082)
"""

import sys
from src.calibrated_mlp_pipeline import run_calibrated_pipeline

def main():
    print("=" * 76)
    print("  CALIBRATED DEEP LEARNING FOR HEART DISEASE RISK PREDICTION")
    print("  Journal: Engineered Science (Manuscript ID: ES-2026-0082.R2)")
    print("=" * 76)

    print("\n[Phase 1] Executing Complete Clinical Calibration Pipeline...")
    results = run_calibrated_pipeline(output_dir="outputs")

    print("\n" + "=" * 76)
    print("  EMPIRICAL EVALUATION RESULTS (Clinical Cohort):")
    print("=" * 76)
    print(f"  • Test Cohort Size            : {results['test_samples']} patients")
    print(f"  • Classification Accuracy     : {results['test_accuracy']*100:.2f}%")
    print(f"  • Raw Uncalibrated Brier Score: {results['uncalibrated_brier_score']:.4f}")
    print(f"  • Calibrated Ensemble Brier   : {results['calibrated_brier_score']:.4f} ({results['brier_improvement_percent']}% error reduction)")
    print(f"  • Cross-Entropy Log Loss      : {results['log_loss']:.4f}")
    print(f"  • ROC-AUC Discrimination      : {results['roc_auc']:.4f}")
    print(f"  • Calibration Reliability Plot: {results['calibration_plot']}")
    print(f"  • ROC-AUC Curve Plot          : {results['roc_plot']}")
    print("=" * 76)
    print("  MULTI-ORGAN DEPENDENCY PIPELINE SCRIPTS:")
    print("  - pipeline/heart.py             : Full NHANES cardio-metabolic risk architecture")
    print("  - pipeline/predict_patient.py   : Patient-level multi-organ risk scoring")
    print("  - pipeline/xai_module.py        : Explainable AI & feature attribution")
    print("=" * 76)

if __name__ == "__main__":
    main()
