#!/usr/bin/env python3

"""
04_clinicalbert_with_synthetic.py

Trains a multi-label ClinicalBERT (Bio_ClinicalBERT) ICD classifier
using synthetic and (optionally) real data.

- Stratified on top-N ICD codes
- Saves all metrics/plots in 'results/' for comparison with previous experiments

Run: python 04_clinicalbert_with_synthetic.py
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             classification_report, hamming_loss, accuracy_score,
                             jaccard_score, multilabel_confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# -------------------- CONFIG -----------------------
os.makedirs("results", exist_ok=True)
SYN_DATA_PATH = "results/synthetic_gpt_icd_data.csv"
# If you have real data, set path here; if not, ignore/comment following line
REAL_DATA_PATH = "DataICD/surgery_reports_icd_multilabel.csv"
N_TOP = 10
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
THRESHOLD = 0.2   # can tune this later!

# -------------- LOAD AND PREP DATA -----------------

print(f"Loading synthetic data: {SYN_DATA_PATH}")
df_syn = pd.read_csv(SYN_DATA_PATH)
df_syn["icd_codes"] = df_syn["icd_codes"].apply(eval)

dfs = [df_syn]

if REAL_DATA_PATH is not None and os.path.exists(REAL_DATA_PATH):
    print(f"Loading real data: {REAL_DATA_PATH}")
    df_real = pd.read_csv(REAL_DATA_PATH)
    df_real["icd_codes"] = df_real["icd_codes"].apply(eval)
    dfs.append(df_real)

df = pd.concat(dfs, ignore_index=True)

all_codes = [c for codes in df["icd_codes"] for c in codes]
code_counts = pd.Series(all_codes).value_counts()
icd_vocab = code_counts.index[:N_TOP].tolist()
print(f"Top-{N_TOP} ICD codes: {icd_vocab}")

def filter_codes(codes):
    return [c for c in codes if c in icd_vocab]
df["icd_codes_filtered"] = df["icd_codes"].apply(filter_codes)
df = df[df["icd_codes_filtered"].map(len) > 0].reset_index(drop=True)

X = df["report_text"]
mlb = MultiLabelBinarizer(classes=icd_vocab)
Y = mlb.fit_transform(df["icd_codes_filtered"])

# ----------- STRATIFIED TRAIN/TEST SPLIT -------------
# If real data present, use ONLY real for test
if REAL_DATA_PATH is not None and os.path.exists(REAL_DATA_PATH):
    train_df = df[df["_merge"] != "real_only"] if "_merge" in df.columns else df_syn
    test_df = df_real
    X_train, Y_train = train_df["report_text"], mlb.transform(train_df["icd_codes_filtered"])
    X_test, Y_test = test_df["report_text"], mlb.transform(test_df["icd_codes_filtered"])
else:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ---------------- BERT DATASET -------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
class ICDTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = np.array(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

train_ds = ICDTextDataset(X_train, Y_train, tokenizer)
test_ds  = ICDTextDataset(X_test,  Y_test,  tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=Y.shape[1], problem_type="multi_label_classification"
)
model = model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    eval_strategy="epoch",
    logging_steps=50,
    save_total_limit=1,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > THRESHOLD).astype(int)
    return {
        'micro_f1': f1_score(labels, preds, average='micro', zero_division=0),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(labels, preds, average='weighted', zero_division=0),
        'micro_precision': precision_score(labels, preds, average='micro', zero_division=0),
        'macro_precision': precision_score(labels, preds, average='macro', zero_division=0),
        'micro_recall': recall_score(labels, preds, average='micro', zero_division=0),
        'macro_recall': recall_score(labels, preds, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(labels, preds),
        'subset_acc': accuracy_score(labels, preds),
        'micro_jaccard': jaccard_score(labels, preds, average='micro', zero_division=0)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

# --------------- TRAIN/VALIDATE -------------------
print("\nTraining and evaluating ClinicalBERT (with synthetic data)..")
trainer.train()

# --------------- EVALUATION -----------------------
model.eval()
probs = []
y_true = []
with torch.no_grad():
    for batch in tqdm(torch.utils.data.DataLoader(test_ds, batch_size=8), desc="Prediction"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        prob = torch.sigmoid(logits).cpu().numpy()
        probs.append(prob)
        y_true.append(labels.cpu().numpy())
probs = np.vstack(probs)
y_true = np.vstack(y_true)
y_pred = (probs > THRESHOLD).astype(int)

# -------------------- METRICS ---------------------
metrics = {
    'micro_f1': f1_score(y_true, y_pred, average='micro'),
    'macro_f1': f1_score(y_true, y_pred, average='macro'),
    'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
    'micro_precision': precision_score(y_true, y_pred, average='micro'),
    'macro_precision': precision_score(y_true, y_pred, average='macro'),
    'micro_recall': recall_score(y_true, y_pred, average='micro'),
    'macro_recall': recall_score(y_true, y_pred, average='macro'),
    'hamming_loss': hamming_loss(y_true, y_pred),
    'subset_acc': accuracy_score(y_true, y_pred),
    'micro_jaccard': jaccard_score(y_true, y_pred, average='micro')
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('results/clinicalbert_metrics.csv', index=False)

report = classification_report(y_true, y_pred, target_names=icd_vocab, zero_division=0)
with open('results/clinicalbert_report.txt', 'w') as f:
    f.write(report)
print("\nClassification report saved to results/clinicalbert_report.txt")

report_dict = classification_report(y_true, y_pred, target_names=icd_vocab, zero_division=0, output_dict=True)
f1s = [report_dict[c]['f1-score'] for c in icd_vocab]
plt.figure(figsize=(8,4))
sns.barplot(x=icd_vocab, y=f1s)
plt.title("ClinicalBERT F1-Score Per ICD Label")
plt.ylabel("F1-Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/clinicalbert_f1_per_label.png", dpi=200)
plt.close()

if len(icd_vocab) <= 10:
    conf_mats = multilabel_confusion_matrix(y_true, y_pred)
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
    plt.suptitle("Label-wise Confusion Matrices (ClinicalBERT)", y=1.02)
    plt.savefig("results/clinicalbert_conf_matrix.png", dpi=200)
    plt.close()

# FINAL SUMMARY
print("\n=== ClinicalBERT (Synthetic Data) Metrics ===")
print(metrics_df.T)
print("Per-label F1-scores:")
for c, f1 in zip(icd_vocab, f1s):
    print(f"{c}: {f1:.3f}")

print("\nAll outputs saved to 'results/'.")