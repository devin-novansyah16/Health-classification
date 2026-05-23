"""
main.py
Full pipeline runner: generate → EDA → train → predict demo
Run from the project root: python main.py
"""

import subprocess
import sys
import os

def run(script: str, label: str):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"[ERROR] {script} exited with code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run("src/generate_data.py", "STEP 1/4 — Generating Dataset")
    run("src/eda.py",           "STEP 2/4 — Exploratory Data Analysis")
    run("src/train.py",         "STEP 3/4 — Training & Evaluating Models")
    run("src/predict.py",       "STEP 4/4 — Demo Predictions")

    print(f"\n{'='*55}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*55}")
    print("  Reports  → reports/")
    print("  Models   → models/")
    print("\n  To predict interactively:")
    print("    python src/predict.py --interactive")
    print(f"{'='*55}\n")
