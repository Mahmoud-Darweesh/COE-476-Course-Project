#!/usr/bin/env python3

"""
Synthetic Surgery Report/ICD Data Generation Script (GPT-4/O)
-------------------------------------------------------------
Generates synthetic surgery reports and realistic ICD-10 labels via OpenAI GPT-4/O API.

USAGE:
    $ export OPENAI_API_KEY=sk-...  # don't store API key in code!
    $ python generate_synthetic_icd_gpt.py

Output: synthetic_gpt_icd_data.csv -- columns: report_text, icd_codes (as Python list-string)

You can adjust the prompt/template and batch settings below.

Author: [Your Name], 2024
"""

import os
import re
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIStatusError

# ------------- Configuration ------------------

MODEL = "gpt-4o"                        # or "gpt-4", or "gpt-3.5-turbo"
N_BATCHES = 100                         # Number of batched API calls
REPORTS_PER_BATCH = 5                   # Samples per batch
SLEEP_BETWEEN_BATCHES = 2               # Seconds to pause between API calls
OUTPUT_FILE = "synthetic_gpt_icd_data.csv"

# ICD code restriction (optional, you may want to constrain codes so you get high-value labels)
# TOP_CODES = ["C10.8", "C32.0", "C02.1", "C13.8", "C32.8"]
# Use in prompt if needed

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
- All content must be fully synthetic, never reference real people.
Generate {n} examples, format EXACTLY as above (repeat with REPORT/ICD_CODES blocks, no numbering, blank lines between samples).
"""

# ------------- Helper Functions ------------------

def parse_gpt_icd_data(raw_text):
    """
    Returns a dataframe: report_text, icd_codes (as list)
    """
    results = []
    for block in re.split(r"REPORT:", raw_text, flags=re.I)[1:]:
        block = block.strip()
        if "ICD_CODES:" not in block: continue
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
            print("Rate limit hit; sleeping.")
            time.sleep(10)
        except APIStatusError as e:
            print(f"API Error (status {e.status_code}); sleeping.")
            time.sleep(10)
        except Exception as ex:
            print(f"OpenAI Error: {ex}, sleeping.")
            time.sleep(15)

# ------------- Main Script -----------------------

if __name__ == "__main__":
    # --- API Key: read from environment for safety ---
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("Set your OpenAI API key in the environment variable OPENAI_API_KEY.")
        exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    all_batches = []
    for batch in tqdm(range(N_BATCHES), desc="GPT-4 batches"):
        gpt_out = generate_gpt_batch(client, n=REPORTS_PER_BATCH)
        df_batch = parse_gpt_icd_data(gpt_out)
        if not df_batch.empty:
            all_batches.append(df_batch)
        print(f"Generated {len(df_batch)} samples in batch {batch+1}.")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    if not all_batches:
        print("No data generated. Check prompt/API settings.")
        exit(2)

    df_full = pd.concat(all_batches, ignore_index=True)
    df_full.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df_full)} synthetic samples to {OUTPUT_FILE}")
    print(df_full.sample(3))  # Quick sample for sanity check