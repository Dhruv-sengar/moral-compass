"""
merge_aita_dataset.py  (fixed)
================================
Reads data/aita deataset.zip directly, handles the exact CSV format:
  Columns: pid, title, post, full post, verdict

Verdict values observed are free-text phrases → normalised to NTA/YTA etc.
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import re
import sys
import zipfile

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE, "data")

DEFAULT_ZIP = os.path.join(DATA_DIR, "aita deataset.zip")
SYNTH_CSV   = os.path.join(DATA_DIR, "synthetic_moral_dataset.csv")
OUT_CSV     = os.path.join(DATA_DIR, "moral_dataset.csv")

# ── Verdict normalisation ──────────────────────────────────────────────────
# Maps any variant → canonical abbreviation → our 3-class label
def normalise_verdict(raw: str) -> str | None:
    """
    Convert any verdict format to our 3-class label.
    Handles: abbreviations, full phrases, mixed case, partial matches.
    """
    if not raw or str(raw).strip().lower() in ("nan", "none", "", "null"):
        return None

    v     = str(raw).strip()
    v_up  = v.upper()
    v_low = v.lower()

    # ── Exact abbreviations ────────────────────────────────────────────
    if v_up in ("NTA", "NAH"):
        return "Ethical"
    if v_up in ("YTA", "ESH"):
        return "Selfish"
    if v_up == "INFO":
        return None

    # ── Dataset-specific exact values ────────────────────────────────────
    # This dataset uses: 'user_ok' and 'user_is_fault'
    EXACT_MAP = {
        "user_ok":          "Ethical",
        "user_is_ok":       "Ethical",
        "user_not_fault":   "Ethical",
        "user_is_fault":    "Selfish",
        "user_at_fault":    "Selfish",
        "user_wrong":       "Selfish",
        "nta":              "Ethical",
        "nah":              "Ethical",
        "yta":              "Selfish",
        "esh":              "Selfish",
        "info":             None,
    }
    if v_low in EXACT_MAP:
        return EXACT_MAP[v_low]

    # ── Substring / phrase matching (case-insensitive) ─────────────────
    ethical_signals = [
        "user_ok", "user_not", "not the a", "not the asshole", "nta", "nah",
        "no assholes here", "not the villain", "not wrong", "not at fault",
        "_ok", "not_fault",
    ]
    selfish_signals = [
        "user_is_fault", "user_fault", "user_at_fault", "user_wrong",
        "you're the a", "youre the a", "yta", "you are the a",
        "everyone sucks", "esh", "asshole", "_fault", "_wrong",
    ]
    info_signals = ["need more info", "not enough info"]

    for sig in info_signals:
        if sig in v_low:
            return None

    for sig in ethical_signals:
        if sig in v_low:
            return "Ethical"

    for sig in selfish_signals:
        if sig in v_low:
            return "Selfish"

    # ── Numeric binary labels (1 = YTA/Selfish, 0 = NTA/Ethical) ─────
    try:
        num = float(v)
        if num == 1.0:
            return "Selfish"
        if num == 0.0:
            return "Ethical"
    except ValueError:
        pass

    return None  # Unknown — drop


# ── Text cleaning ──────────────────────────────────────────────────────────
_AITA_RE = re.compile(r"(?i)(am i (the )?(a[s*]+hole|aita)\s*[\?,]?\s*|aita\s+for\s+)")

def clean(text: str) -> str:
    text = str(text)
    text = _AITA_RE.sub("", text)
    text = re.sub(r"\*\*|~~|__", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── Load & parse an AITA dataframe ────────────────────────────────────────
def parse_aita_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    print(f"    Columns : {list(df.columns)}")

    # ── Pick best text column ──────────────────────────────────────────
    # Prefer longer content: 'full post' > 'post' > 'title' > ...
    TEXT_PREF = ["full post", "post", "body", "selftext", "text", "title"]
    text_col = None
    for pref in TEXT_PREF:
        if pref in cols_lower:
            text_col = cols_lower[pref]
            break
    if text_col is None:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        text_col = max(
            str_cols,
            key=lambda c: df[c].dropna().str.len().mean() if df[c].dropna().shape[0] else 0
        )
    print(f"    Text col: '{text_col}'")

    # ── Pick verdict column ────────────────────────────────────────────
    VERDICT_PREF = ["verdict", "label", "flair", "link_flair_text",
                    "post_flair", "flair_text", "verdict_label"]
    label_col = None
    for pref in VERDICT_PREF:
        if pref in cols_lower:
            label_col = cols_lower[pref]
            break
    if label_col is None:
        print(f"    [WARN] No verdict column found in '{source}'. Skipping.")
        return pd.DataFrame(columns=["text", "label"])
    print(f"    Label col: '{label_col}'")

    # ── Show sample verdict values ────────────────────────────────────
    sample_verdicts = df[label_col].dropna().unique()[:20]
    print(f"    Sample verdicts: {list(sample_verdicts)}")

    # ── Build output ──────────────────────────────────────────────────
    out = pd.DataFrame({
        "text":        df[text_col].astype(str),
        "raw_verdict": df[label_col].astype(str).str.strip(),
    })

    out["label"] = out["raw_verdict"].apply(normalise_verdict)

    before = len(out)
    out    = out.dropna(subset=["label"])
    print(f"    Kept   : {len(out):,} / {before:,} rows after verdict mapping")

    if len(out) > 0:
        dist = out["label"].value_counts()
        print(f"    Dist   : {dist.to_dict()}")

    # Clean text & drop very short entries
    out["text"] = out["text"].apply(clean)
    out = out[out["text"].str.len() >= 30]

    return out[["text", "label"]].reset_index(drop=True)


# ── Read a CSV from inside a zip ──────────────────────────────────────────
def read_csv_from_zip(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with z.open(name) as f:
        raw = f.read()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"Cannot decode {name}")


# ── Main ──────────────────────────────────────────────────────────────────
def main(zip_path: str | None = None, csv_path: str | None = None) -> None:
    frames: list[pd.DataFrame] = []

    # ── 1. AITA data ──────────────────────────────────────────────────────
    if csv_path and os.path.exists(csv_path):
        print(f"\n[1] Loading AITA CSV: {csv_path}")
        aita = parse_aita_df(pd.read_csv(csv_path, low_memory=False), csv_path)
        if len(aita):
            frames.append(aita)

    else:
        if zip_path is None:
            zip_path = DEFAULT_ZIP if os.path.exists(DEFAULT_ZIP) else None
            if zip_path is None:
                zips = glob.glob(os.path.join(DATA_DIR, "*.zip"))
                zip_path = zips[0] if zips else None

        if zip_path and os.path.exists(zip_path):
            print(f"\n[1] Reading zip: {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                print(f"    CSVs inside: {csv_names}")
                for name in csv_names:
                    print(f"\n    → {name}")
                    try:
                        raw_df = read_csv_from_zip(z, name)
                        aita   = parse_aita_df(raw_df, name)
                        if len(aita):
                            frames.append(aita)
                            print(f"    ✓ Added {len(aita):,} rows from {name}")
                    except Exception as e:
                        print(f"    [ERROR] {e}")
        else:
            print("\n[1] No AITA zip found — using synthetic data only.")

    # ── 2. Synthetic data ─────────────────────────────────────────────────
    print(f"\n[2] Loading synthetic data: {SYNTH_CSV}")
    if os.path.exists(SYNTH_CSV):
        s = pd.read_csv(SYNTH_CSV)
        if "label" not in s.columns and "true_label" in s.columns:
            s = s.rename(columns={"true_label": "label"})
        s = s[["text", "label"]].dropna()
        print(f"    {len(s):,} rows — {s['label'].value_counts().to_dict()}")
        frames.append(s)
    else:
        print("    [WARN] synthetic_moral_dataset.csv not found!")

    if not frames:
        print("\n[ERROR] No data loaded. Exiting.")
        sys.exit(1)

    # ── 3. Merge & deduplicate ────────────────────────────────────────────
    print(f"\n[3] Merging {len(frames)} source(s) …")
    merged = pd.concat(frames, ignore_index=True)

    # Ensure 'label' column exists
    if "label" not in merged.columns:
        print("[ERROR] 'label' column missing after concat!")
        sys.exit(1)

    merged["_key"] = merged["text"].str.lower().str.strip()
    before = len(merged)
    merged = merged.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    merged = merged.dropna(subset=["label"])
    print(f"    {before:,} → {len(merged):,} rows after dedup")
    print(f"    Label dist: {merged['label'].value_counts().to_dict()}")

    # Shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── 4. Balance (cap majority at 3× minority) ──────────────────────────
    print(f"\n[4] Balancing classes …")
    counts  = merged["label"].value_counts()
    min_cnt = int(counts.min())
    cap     = min_cnt * 3

    # groupby/apply can make 'label' the index in some pandas versions
    # — build balanced df manually to avoid KeyError
    parts = []
    for lbl, grp in merged.groupby("label"):
        parts.append(grp.sample(min(len(grp), cap), random_state=42))
    balanced = (
        pd.concat(parts, ignore_index=True)
          .sample(frac=1, random_state=42)
          .reset_index(drop=True)
    )

    print(f"    After balancing: {balanced['label'].value_counts().to_dict()}")

    # ── 5. Save ───────────────────────────────────────────────────────────
    balanced["true_label"] = balanced["label"]
    balanced["weak_label"] = balanced["label"]
    out_df = balanced[["text", "true_label", "weak_label"]]

    os.makedirs(DATA_DIR, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(out_df):,} rows → {OUT_CSV}")
    print(f"       Distribution:\n{out_df['true_label'].value_counts().to_string()}")
    print("\n✅  Next: python src/train.py")


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", metavar="PATH",
                    help='Path to AITA zip (default: data/aita deataset.zip)')
    ap.add_argument("--csv", metavar="PATH",
                    help="Path to already-extracted AITA CSV")
    args = ap.parse_args()
    main(zip_path=args.zip, csv_path=args.csv)
