#!/usr/bin/env python3

"""
03_gpt_synthetic_data.py

Generates synthetic surgery reports plus ICD-10 codes using the OpenAI GPT-4o API.
All output is saved in 'results/synthetic_gpt_icd_data.csv'.

Usage:
    export OPENAI_API_KEY=sk-YOURKEYHERE
    python 03_gpt_synthetic_data.py

Settings can be changed at the top of the script.
"""

import os
import re
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIStatusError

# ======================= USER SETTINGS ==============================
MODEL = "gpt-4o"
N_BATCHES = 100              # Each API call is a batch (e.g. 100 batches × 5 = 500 samples)
REPORTS_PER_BATCH = 5
SLEEP_BETWEEN_BATCHES = 2    # seconds
OUTPUT_CSV = "results/synthetic_gpt_icd_data.csv"
os.makedirs("results", exist_ok=True)
# ====================================================================

TEMPLATE = """
You are a medical data generator. Produce {n} synthetic head and neck cancer surgery reports, each followed by a realistic set of ICD-10 diagnosis codes.
Format for each example:
REPORT: <surgical narrative, at least a paragraph, rich clinical detail>
ICD_CODES: <relevant ICD-10 codes, comma separated, e.g. C32.0, C10.8, C77.0>

Guidelines:
- Use diverse tumor sites (e.g. larynx, oropharynx, hypopharynx, oral cavity, etc.)
- Sometimes include lymph node metastases (e.g. C77.0)
- Assign 1-4 ICD codes per case, all codes must fit the narrative!
- Mix simple and advanced cases.
- All content must be fully synthetic, never real or repeating real examples.
Now generate {n} examples, each in the format above, with a blank line separating examples.
"""

def parse_gpt_icd_data(raw_text):
    """
    Returns a dataframe: report_text (str), icd_codes (list of str)
    """
    results = []
    for block in re.split(r"REPORT:", raw_text, flags=re.I)[1:]:
        block = block.strip()
        if "ICD_CODES:" not in block:
            continue
        try:
            report, icd_codes = block.split("ICD_CODES:", 1)
            report_text = report.strip().replace('\n', ' ')
            codes = [c for c in re.findall(r"[A-Z]\d{2}\.?[A-Z\d]*", icd_codes)]
            if codes:
                results.append({"report_text": report_text, "icd_codes": codes})
        except Exception:
            continue
    return pd.DataFrame(results)

def generate_gpt_batch(client, n=5, model=MODEL):
    prompt = TEMPLATE.format(n=n)
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1800
            )
            return completion.choices[0].message.content
        except RateLimitError:
            print("Rate limit hit; sleeping 10s.")
            time.sleep(10)
        except APIStatusError as e:
            print(f"API error {e.status_code}; sleeping 10s.")
            time.sleep(10)
        except Exception as ex:
            print(f"OpenAI Exception: {ex}; sleeping 15s.")
            time.sleep(15)

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please export your OpenAI key as OPENAI_API_KEY.")
        exit(1)

    client = OpenAI(api_key=api_key)
    all_batches = []

    for batch in tqdm(range(N_BATCHES), desc=f"GPT-4o ({REPORTS_PER_BATCH}x per batch)"):
        gpt_text = generate_gpt_batch(client, n=REPORTS_PER_BATCH)
        df_batch = parse_gpt_icd_data(gpt_text)
        if not df_batch.empty:
            all_batches.append(df_batch)
        print(f"Batch {batch+1:>3}: Generated {len(df_batch)} samples.")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    if not all_batches:
        print("No data generated (empty result). Check prompt/API settings.")
        exit(2)

    df_full = pd.concat(all_batches, ignore_index=True)
    df_full.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df_full)} synthetic samples to {OUTPUT_CSV}")
    print("\nSample rows:\n", df_full.sample(min(5, len(df_full))))

    # Optional: ICD label frequency stats
    all_codes = [c for codes in df_full["icd_codes"].apply(eval if isinstance(df_full["icd_codes"].iloc[0], str) else lambda x: x) for c in codes]
    code_counts = pd.Series(all_codes).value_counts()
    print("\nTop 10 ICD codes in generated data:\n", code_counts.head(10))
    code_counts.to_csv("results/synthetic_icd_label_freq.csv")