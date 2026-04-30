"""
run_pipeline.py  –  One-shot: merge AITA data → train model → done
===================================================================
Run from the project root:
    python run_pipeline.py

Steps
-----
  1. Merge AITA zip + synthetic data  →  data/moral_dataset.csv
  2. Train LR + SVM models            →  models/best_model.pkl
"""

import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)

run([sys.executable, "src/merge_aita_dataset.py"])
run([sys.executable, "src/train.py"])

print("\n" + "="*60)
print("  ✅ Pipeline complete!")
print("  Start backend: uvicorn backend.main:app --reload --port 8000")
print("="*60)
