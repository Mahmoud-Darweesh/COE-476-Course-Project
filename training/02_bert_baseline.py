#!/usr/bin/env python3

"""
02_bert_baseline.py

Runs a ClinicalBERT multi-label classifier baseline on synthetic ICD-labeled surgery reports.
Saves all metrics and plots for analysis.

Run: python 02_bert_baseline.py
Data: DataICD/synthetic_gpt_icd_data.csv

Outputs (all in results/):
    bert_report.txt, bert_metrics.csv, bert_f1_per_label.png,
    bert_conf_matrix.png, etc.
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

# -------------------- SETUP RESULTS FOLDER --------------------
os.makedirs("results", exist_ok=True)

# -------------------- LOAD AND EXPLORE DATA -------------------
DATA_PATH = "DataICD/surgery_reports_icd_multilabel.csv"
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
df['icd_codes'] = df['icd_codes'].apply(eval)

all_codes = [c for codes in df['icd_codes'] for c in codes]
code_counts = pd.Series(all_codes).value_counts()
N_TOP = 10
icd_vocab = code_counts.index[:N_TOP].tolist()

# Filter data to only samples with top N codes
def filter_codes(codes):
    return [c for c in codes if c in icd_vocab]

df['icd_codes_filtered'] = df['icd_codes'].apply(filter_codes)
df = df[df['icd_codes_filtered'].map(len) > 0].reset_index(drop=True)

print(f"Filtered dataset has {len(df)} samples; using codes: {icd_vocab}")

X = df['report_text']
mlb = MultiLabelBinarizer(classes=icd_vocab)
Y = mlb.fit_transform(df['icd_codes_filtered'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ----------- PLOT LABEL FREQUENCY BEFORE TRAINING -------------
code_counts = pd.Series([c for codes in df['icd_codes_filtered'] for c in codes]).value_counts()
plt.figure(figsize=(10,4))
sns.barplot(x=code_counts.index[:20], y=code_counts.values[:20])
plt.title("ICD Label Frequency (Top 20)")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/label_freqs_bert.png", dpi=200)
plt.close()

# -------------------- BERT Multi-Label Classification --------------------

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # or "bert-base-uncased" for standard test

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
class ICDTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

train_ds = ICDTextDataset(X_train, Y_train, tokenizer)
test_ds  = ICDTextDataset(X_test,  Y_test,  tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=Y.shape[1],
    problem_type="multi_label_classification"
)
model = model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    eval_strategy="steps",
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
    preds = (probs > 0.2).astype(int)  # threshold can be tuned
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

# ------------- TRAIN & EVALUATE ----------------
print("\nTraining and evaluating ClinicalBERT..")
trainer.train()

# -- On Eval set, get detailed predictions --
print("Inference & saving results...")
probs = []
y_true = []
model.eval()
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
y_pred = (probs > 0.2).astype(int)

# -------------------- METRICS --------------------
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
metrics_df.to_csv('results/bert_metrics.csv', index=False)

## Classification report text
report = classification_report(y_true, y_pred, target_names=icd_vocab, zero_division=0)
with open('results/bert_report.txt', 'w') as f:
    f.write(report)
print("\nClassification report saved to results/bert_report.txt")

## Per-label F1 Score barplot
report_dict = classification_report(y_true, y_pred, target_names=icd_vocab, zero_division=0, output_dict=True)
f1s = [report_dict[c]['f1-score'] for c in icd_vocab]
plt.figure(figsize=(8,4))
sns.barplot(x=icd_vocab, y=f1s)
plt.title("BERT Baseline F1-Score Per ICD Label")
plt.ylabel("F1-Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/bert_f1_per_label.png", dpi=200)
plt.close()

# ---- Confusion matrices, if N_TOP <= 10 ----
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
    plt.suptitle("Label-wise Confusion Matrices (BERT)", y=1.02)
    plt.savefig("results/bert_conf_matrix.png", dpi=200)
    plt.close()

print("\n==== BERT Baseline Metrics ====")
print(metrics_df.T)
print("Per-label F1-scores:")
for c, f1 in zip(icd_vocab, f1s):
    print(f"{c}: {f1:.3f}")

print("\nAll outputs saved to 'results/'.")
