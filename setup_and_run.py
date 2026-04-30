#!/usr/bin/env python3
"""
setup_and_run.py  –  Complete Moral Compass Classifier setup
=============================================================
Double-click this OR run from VS Code terminal:

    cd "D:\PROJECTS\MORAL COMPASS"
    .venv\Scripts\activate
    python setup_and_run.py

What it does
------------
  1. Installs Python dependencies
  2. Generates synthetic dataset (if needed)
  3. Merges AITA zip → data/moral_dataset.csv
  4. Trains ML models → models/best_model.pkl
  5. Prints instructions to start backend + frontend
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable


def run(cmd: list, check: bool = True):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    print("-" * 60)
    r = subprocess.run(cmd, cwd=ROOT)
    if check and r.returncode != 0:
        print(f"\n[FAIL] Exit code {r.returncode}")
        sys.exit(r.returncode)
    return r.returncode


# ── 1. Dependencies ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — Installing Python dependencies")
print("="*60)
run([PY, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

# ── 2. Synthetic dataset ──────────────────────────────────────────────────
synth_csv = os.path.join(ROOT, "data", "synthetic_moral_dataset.csv")
if not os.path.exists(synth_csv):
    print("\n" + "="*60)
    print("  STEP 2 — Generating synthetic dataset")
    print("="*60)
    run([PY, "src/data_generation.py"])
else:
    print(f"\n[SKIP] synthetic_moral_dataset.csv already exists")

# ── 3. Merge AITA data ────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — Merging AITA dataset with synthetic data")
print("="*60)
run([PY, "src/merge_aita_dataset.py"])

# ── 4. Train models ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 — Training ML models")
print("="*60)
run([PY, "src/train.py"])

# ── Done ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ✅  Setup complete!")
print("="*60)
print(r"""
Now open TWO terminal windows:

  Terminal 1 — Backend (FastAPI):
  ─────────────────────────────────────────────────────────
  cd "D:\PROJECTS\MORAL COMPASS"
  .venv\Scripts\activate
  uvicorn backend.main:app --reload --port 8000

  API docs → http://localhost:8000/docs

  Terminal 2 — Frontend (React):
  ─────────────────────────────────────────────────────────
  cd "D:\PROJECTS\MORAL COMPASS\frontend"
  npm run dev

  App      → http://localhost:5173
""")
