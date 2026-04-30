"""
inspect_zip.py – Quick script to peek inside the AITA zip and show columns.
Run: python src/inspect_zip.py
"""
import zipfile, os, io
import pandas as pd

ZIP_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aita deataset.zip")

print(f"Opening: {ZIP_PATH}\n")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    names = z.namelist()
    print("Files inside zip:")
    for n in names:
        info = z.getinfo(n)
        print(f"  {n}  ({info.file_size:,} bytes)")
    
    print()
    for n in names:
        if n.endswith(".csv"):
            print(f"\n--- Inspecting: {n} ---")
            with z.open(n) as f:
                try:
                    df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), nrows=3)
                    print("Columns:", df.columns.tolist())
                    print(df.to_string())
                except Exception as e:
                    print(f"  Error: {e}")
