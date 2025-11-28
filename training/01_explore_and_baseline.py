#!/usr/bin/env python3

"""
01_explore_and_baseline.py

Loads synthetic ICD-labeled surgery reports, shows dataset info,
and runs a multi-label TF-IDF + Logistic Regression baseline.
Saves all metrics and plots.

Run: python 01_explore_and_baseline.py
Dataset: DataICD/synthetic_gpt_icd_data.csv

Outputs:
    results/label_freqs.png           -- Label histogram
    results/baseline_report.txt       -- Full classification report
    results/baseline_metrics.csv      -- Main metrics table
    results/baseline_f1_per_label.png -- F1-score barplot per label
    results/conf_matrix.png           -- Multilabel confusion matrix heatmaps (if n_labels <= 10)
    results/...

"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             classification_report, hamming_loss, accuracy_score,
                             jaccard_score, multilabel_confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -------------------- SETUP RESULTS FOLDER --------------------
os.makedirs("results", exist_ok=True)

# -------------------- LOAD AND EXPLORE DATA -------------------
DATA_PATH = "DataICD/surgery_reports_icd_multilabel.csv"
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
df['icd_codes'] = df['icd_codes'].apply(eval)

print("Number of samples:", len(df))
all_codes = [c for codes in df['icd_codes'] for c in codes]
code_counts = pd.Series(all_codes).value_counts()

plt.figure(figsize=(10,4))
sns.barplot(x=code_counts.index[:20], y=code_counts.values[:20])
plt.title("ICD Label Frequency (Top 20)")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/label_freqs.png", dpi=200)
plt.close()

print("\nICD code distribution (top 10):\n", code_counts.head(10))

N_TOP = 5
icd_vocab = code_counts.index[:N_TOP].tolist()

# Filter to top labels for baseline
def filter_codes(codes):
    return [c for c in codes if c in icd_vocab]
df['icd_codes_filtered'] = df['icd_codes'].apply(filter_codes)
df = df[df['icd_codes_filtered'].map(len) > 0].reset_index(drop=True)
print(f"\nFiltered dataset (at least one of top {N_TOP} codes): {len(df)} samples")
print("ICD code list used:", icd_vocab)

# -------------------- SPLIT & PREPARE DATA --------------------
X = df['report_text']
mlb = MultiLabelBinarizer(classes=icd_vocab)
Y = mlb.fit_transform(df['icd_codes_filtered'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
Xtr_tfidf = tfidf.fit_transform(X_train)
Xte_tfidf = tfidf.transform(X_test)

model = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=3, solver="liblinear", random_state=42))
model.fit(Xtr_tfidf, Y_train)
Y_pred = model.predict(Xte_tfidf)

# -------------------- METRICS CALCULATION ---------------------
# Main summary metrics
metrics = {
    'micro_f1': f1_score(Y_test, Y_pred, average='micro'),
    'macro_f1': f1_score(Y_test, Y_pred, average='macro'),
    'weighted_f1': f1_score(Y_test, Y_pred, average='weighted'),
    'micro_precision': precision_score(Y_test, Y_pred, average='micro'),
    'macro_precision': precision_score(Y_test, Y_pred, average='macro'),
    'micro_recall': recall_score(Y_test, Y_pred, average='micro'),
    'macro_recall': recall_score(Y_test, Y_pred, average='macro'),
    'hamming_loss': hamming_loss(Y_test, Y_pred),
    'subset_acc': accuracy_score(Y_test, Y_pred),
    'micro_jaccard': jaccard_score(Y_test, Y_pred, average='micro')
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('results/baseline_metrics.csv', index=False)

# Per-class (label) F1s/support
report = classification_report(Y_test, Y_pred, target_names=icd_vocab, zero_division=0, output_dict=True)
report_text = classification_report(Y_test, Y_pred, target_names=icd_vocab, zero_division=0)
with open('results/baseline_report.txt', 'w') as f:
    f.write(report_text)
print("\nWrote detailed classification report to results/baseline_report.txt")

f1s = [report[c]['f1-score'] for c in icd_vocab]
supports = [report[c]['support'] for c in icd_vocab]

plt.figure(figsize=(8,4))
sns.barplot(x=icd_vocab, y=f1s)
plt.title("Baseline F1-Score Per ICD Label")
plt.ylabel("F1-Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/baseline_f1_per_label.png", dpi=200)
plt.close()

# ----------------- MULTILABEL CONFUSION MATRICES -------------
if len(icd_vocab) <= 10 and Y_test.shape[1] <= 10:
    conf_mats = multilabel_confusion_matrix(Y_test, Y_pred)
    fig, axs = plt.subplots(len(icd_vocab), 1, figsize=(5, 2*len(icd_vocab)), constrained_layout=True)
    if len(icd_vocab) == 1:
        axs = [axs]
    for i, code in enumerate(icd_vocab):
        ax = axs[i]
        cm = conf_mats[i]
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cbar=False,
                    xticklabels=["Not "+code, code], yticklabels=["Not "+code, code])
        ax.set_title(f"Confusion for {code}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.suptitle("Label-wise Confusion Matrices (TFIDF+LR)", y=1.02)
    plt.savefig("results/conf_matrix.png", dpi=200)
    plt.close()

# ----------------- SAVE LABEL BINARIZER (OPTIONAL) -----------
pickle.dump(icd_vocab, open("results/icd_vocab_top10.pkl", "wb"))

# ------------ PRINT SUMMARY TO SCREEN ------------------------
print("\n=== Baseline Results: TF-IDF + Logistic Regression ===")
print(metrics_df.T)
print("Per-label F1-scores:")
for c, f1 in zip(icd_vocab, f1s):
    print(f"{c}: {f1:.3f}")

print("\nPlots and all metrics/results saved in 'results/' folder.")
