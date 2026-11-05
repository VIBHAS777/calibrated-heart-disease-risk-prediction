"""
run_pipeline.py
===============
Master orchestration script for the Multi-Organ Crosstalk Risk Pipeline.
Runs self-contained organ pipelines with timing, validation, and consolidated reporting.

Usage:
  python pipeline/run_pipeline.py                     # Runs all 5 organs
  python pipeline/run_pipeline.py --organ heart        # Runs only Heart
  python pipeline/run_pipeline.py --organ kidney       # Runs only Kidney
  python pipeline/run_pipeline.py --organ liver        # Runs only Liver
  python pipeline/run_pipeline.py --organ pancreas     # Runs only Pancreas
  python pipeline/run_pipeline.py --organ adipose      # Runs only Adipose
"""

import os
import sys
import time
import subprocess
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(REPO_ROOT, "pipeline")

ORGANS = [
    ("heart.py",    "Heart Organ Pipeline (Hypertension Risk)"),
    ("kidney.py",   "Kidney Organ Pipeline (Albuminuria / CKD Risk)"),
    ("liver.py",    "Liver Organ Pipeline (MASLD / Steatosis Risk)"),
    ("pancreas.py", "Pancreas Organ Pipeline (Adenocarcinoma Risk)"),
    ("adipose.py",  "Adipose Organ Pipeline (Visceral Adiposity Risk)"),
]

def run_organ_file(script_name: str, description: str) -> bool:
    script_path = os.path.join(PIPELINE_DIR, script_name)
    print("\n" + "=" * 75)
    print(f" EXECUTING: {script_name} — {description}")
    print("=" * 75)
    t0 = time.time()
    res = subprocess.run([sys.executable, script_path], cwd=REPO_ROOT)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n [ERROR] {script_name} failed with code {res.returncode} after {elapsed:.1f}s!")
        return False
    print(f"\n [SUCCESS] Completed {script_name} in {elapsed:.1f}s.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Multi-Organ Risk Pipeline Runner")
    parser.add_argument(
        "--organ", type=str, default=None,
        choices=["heart", "kidney", "liver", "pancreas", "adipose"],
        help="Run only a specific organ pipeline"
    )
    args = parser.parse_args()

    print("=" * 75)
    print(" MULTI-ORGAN RISK PIPELINE — MASTER ORCHESTRATOR")
    print(f" Repository Root: {REPO_ROOT}")
    print("=" * 75)

    if args.organ:
        target = f"{args.organ.lower()}.py"
        matched = [(f, desc) for f, desc in ORGANS if f == target]
        if not matched:
            print(f"Unknown organ: {args.organ}")
            sys.exit(1)
        script, desc = matched[0]
        success = run_organ_file(script, desc)
        if not success:
            sys.exit(1)
    else:
        print(f"Running all {len(ORGANS)} self-contained organ pipelines sequentially...")
        start_time = time.time()
        for script, desc in ORGANS:
            ok = run_organ_file(script, desc)
            if not ok:
                print(f"\n Pipeline stopped due to error in {script}.")
                sys.exit(1)
        total_time = time.time() - start_time
        print("\n" + "=" * 75)
        print(f" ALL 5 ORGAN PIPELINES COMPLETED SUCCESSFULLY IN {total_time:.1f}s!")
        print(f" Results & figures are available in: {os.path.join(PIPELINE_DIR, 'outputs')}")
        print("=" * 75)

        # Generate Master Cross-Organ Disease Correlation Heatmaps
        run_organ_file("plot_cross_organ_disease_correlation.py", "Cross-Organ Disease Correlation Heatmaps")
        print("\n" + "=" * 75)
        print(" MASTER PIPELINE & CORRELATION HEATMAPS EXECUTION COMPLETE")
        print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
