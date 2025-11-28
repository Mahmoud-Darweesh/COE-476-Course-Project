#!/usr/bin/env python3

"""
Data Preparation Script for ICD Coding of Surgery Reports
---------------------------------------------------------

This script assembles a dataset for multi-label ICD coding of head and neck
surgery reports. It expects a folder structure where each patient's text files
(respective sections and ICD annotation) are present.

Outputs a CSV file with one row per patient:
    - patient_id
    - report_text
    - description_text
    - history_text
    - icd_codes (as a stringified list)

USAGE:
    python data_prep.py

Adjust the folder paths below if your data is in different locations.
"""

import os
import re
import pandas as pd
from glob import glob
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def extract_icd_codes_from_text(text):
    """
    Extracts all ICD codes from a string,
    e.g.: 'Hypopharynxkarzinom[C13.9 ] Larynxkarzinom[C32.9 ] Halslymphknotenmetastasen[C77.0 B]'
    Returns a list like: ['C13.9', 'C32.9', 'C77.0B']
    """
    if not isinstance(text, str):
        return []
    return [c.replace(" ", "") for c in re.findall(r'\[([A-Z0-9\.\- ]+)\]', text) if c.strip()]

def load_text_or_empty(fname):
    """
    Reads file, returns text or empty string if missing.
    """
    try:
        with open(fname, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def get_patient_ids_from_reports(report_dir):
    """
    Gets all patient IDs from the report filenames (expects pattern: ..._<ID>.txt)
    """
    files = glob(os.path.join(report_dir, "*.txt"))
    ids = []
    for f in files:
        m = re.search(r'_(\d+)', os.path.basename(f))
        if m:
            ids.append(m.group(1))
    return sorted(set(ids))

# ------------------------------------------------------------------------------
# Main Data Assembly Function
# ------------------------------------------------------------------------------

def build_icd_dataset(report_dir, icd_dir, descr_dir, hist_dir):
    """
    Returns a DataFrame:
    patient_id | report_text | description_text | history_text | icd_codes (list)
    """
    patient_ids = get_patient_ids_from_reports(report_dir)
    records = []

    for pid in tqdm(patient_ids, desc="Processing patients"):
        report_f = os.path.join(report_dir, f"SurgeryReport_{pid}.txt")
        icd_f    = os.path.join(icd_dir, f"SurgeryReport_ICD_Codes_{pid}.txt")
        descr_f  = os.path.join(descr_dir, f"SurgeryDescriptionEnglish_{pid}.txt")
        hist_f   = os.path.join(hist_dir, f"SurgeryReport_History_{pid}.txt")

        report_text = load_text_or_empty(report_f)
        icd_text    = load_text_or_empty(icd_f)
        descr_text  = load_text_or_empty(descr_f)
        hist_text   = load_text_or_empty(hist_f)

        icd_codes = extract_icd_codes_from_text(icd_text)
        records.append({
            "patient_id": pid,
            "report_text": report_text,
            "description_text": descr_text,
            "history_text": hist_text,
            "icd_codes": icd_codes
        })

    df = pd.DataFrame(records)
    return df

# ------------------------------------------------------------------------------
# Script Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # --------- PATHS (edit as needed) ----------------------
    report_dir = "TextData/reports_english/"
    icd_dir    = "TextData/icd_codes/"
    descr_dir  = "TextData/surgery_descriptions_english/"
    hist_dir   = "TextData/histories_english/"
    out_csv    = "surgery_reports_icd_multilabel.csv"

    # --------- Build dataset -------------------------------
    df = build_icd_dataset(
        report_dir=report_dir,
        icd_dir=icd_dir,
        descr_dir=descr_dir,
        hist_dir=hist_dir
    )

    # --------- Basic Dataset Stats -------------------------
    print("Example rows:\n", df.head(3).T)
    print("\nICD code statistics:")
    all_icds = [icd for codes in df['icd_codes'] for icd in codes]
    print(pd.Series(all_icds).value_counts().head(10))

    # --------- Save as CSV --------------------------------
    # Save icd_codes as stringified list (for later eval())
    df['icd_codes'] = df['icd_codes'].apply(str)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}")