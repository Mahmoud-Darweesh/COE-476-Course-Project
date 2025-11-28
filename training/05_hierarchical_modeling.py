#!/usr/bin/env python3

"""
05_hierarchical_modeling.py

Hierarchical ICD coding with ClinicalBERT.
Predicts both parent (ICD "site") and child (full ICD string) codes,
using a hierarchical loss.

All metrics, reports, and plots are saved in the results/ folder.

Run: python 05_hierarchical_modeling.py

Dependencies: torch, transformers, pandas, numpy, scikit-learn, matplotlib, seaborn, tqdm
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             classification_report, multilabel_confusion_matrix,
                             hamming_loss, accuracy_score, jaccard_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, Trainer, TrainingArguments

# ----------- SETTINGS & FILE PATHS --------------------------
os.makedirs("results", exist_ok=True)
SYN_DATA_PATH = "results/synthetic_gpt_icd_data.csv"
N_TOP = 10
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
THRESH_PARENT = 0.3
THRESH_CHILD = 0.2

# Optionally point to real CSV to use as test
REAL_DATA_PATH = "DataICD/surgery_reports_icd_multilabel.csv"
# ------------------------------------------------------------

def get_parent(code):
    # Take the 4-character site prefix (e.g., "C32.1" ⇒ "C32.")
    return code[:4] if len(code) > 4 else code[:3]

# ---------------- LOAD & PREP DATA --------------------------
print("Loading synthetic data:", SYN_DATA_PATH)
df_syn = pd.read_csv(SYN_DATA_PATH)
df_syn["icd_codes"] = df_syn["icd_codes"].apply(eval)

dfs = [df_syn]
if REAL_DATA_PATH and os.path.exists(REAL_DATA_PATH):
    print("Loading real data:", REAL_DATA_PATH)
    df_real = pd.read_csv(REAL_DATA_PATH)
    df_real["icd_codes"] = df_real["icd_codes"].apply(eval)
    dfs.append(df_real)

df = pd.concat(dfs, ignore_index=True)

all_codes = [c for codes in df["icd_codes"] for c in codes]
code_counts = pd.Series(all_codes).value_counts()
child_codes = code_counts.index[:N_TOP].tolist()
df["icd_codes_filtered"] = df["icd_codes"].apply(lambda xs: [c for c in xs if c in child_codes])
df = df[df["icd_codes_filtered"].map(len) > 0].reset_index(drop=True)

print(f"Using {len(child_codes)} child codes:\n", child_codes)

df["parent_codes"] = df["icd_codes_filtered"].apply(lambda xs: list({get_parent(c) for c in xs}))
all_parent_codes = sorted(set([p for lst in df["parent_codes"] for p in lst]))
print("Parent codes (site groups):", all_parent_codes)

parent_mlb = MultiLabelBinarizer(classes=all_parent_codes)
child_mlb = MultiLabelBinarizer(classes=child_codes)

Y_parent = parent_mlb.fit_transform(df["parent_codes"])
Y_child = child_mlb.fit_transform(df["icd_codes_filtered"])

X = df["report_text"]

X_train, X_test, Y_parent_train, Y_parent_test, Y_child_train, Y_child_test = train_test_split(
    X, Y_parent, Y_child, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

# ----------------- DATASET CLASS ---------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
class HierICDDataset(Dataset):
    def __init__(self, texts, y_parents, y_children, tokenizer, max_len=256):
        self.texts = list(texts)
        self.y_parent = np.asarray(y_parents, dtype=np.float32)
        self.y_child = np.asarray(y_children, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels_parent"] = torch.tensor(self.y_parent[idx], dtype=torch.float32)
        item["labels_child"] = torch.tensor(self.y_child[idx], dtype=torch.float32)
        return item

train_ds = HierICDDataset(X_train, Y_parent_train, Y_child_train, tokenizer)
test_ds  = HierICDDataset(X_test,  Y_parent_test,  Y_child_test,  tokenizer)

# -------- HIERARCHICAL BERT MODEL (PARENT+CHILD) ---------
class HierClinicalBERT(BertPreTrainedModel):
    def __init__(self, config, num_parents, num_children, hier_loss_weight=0.7):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(0.2)
        self.parent_head = torch.nn.Linear(config.hidden_size, num_parents)
        self.child_head  = torch.nn.Linear(config.hidden_size, num_children)
        self.hier_loss_weight = hier_loss_weight
        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels_parent=None,
        labels_child=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.dropout(outputs.pooler_output if hasattr(outputs, "pooler_output")
                                    else outputs.last_hidden_state[:, 0])
        logits_parent = self.parent_head(pooled_output)
        logits_child  = self.child_head(pooled_output)
        out = {"logits_parent": logits_parent, "logits_child": logits_child}
        if labels_parent is not None and labels_child is not None:
            # Parent loss
            loss_parent = torch.nn.functional.binary_cross_entropy_with_logits(logits_parent, labels_parent)
            # Child: only mask those with predicted parent
            loss_child_all = torch.nn.functional.binary_cross_entropy_with_logits(logits_child, labels_child, reduction='none')
            with torch.no_grad():
                mask = torch.sigmoid(logits_parent) > THRESH_PARENT
                child2parent = [get_parent(cc) for cc in child_codes]
                parent_mask = torch.stack(
                    [mask[:, all_parent_codes.index(p)] if p in all_parent_codes else mask[:, 0]*0 for p in child2parent],
                    dim=1
                )
            weighted_loss_child = (parent_mask.float() * loss_child_all +
                                   (1 - parent_mask.float()) * 0.5 * loss_child_all).mean()
            total_loss = self.hier_loss_weight * loss_parent + (1 - self.hier_loss_weight) * weighted_loss_child
            out["loss"] = total_loss
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierClinicalBERT.from_pretrained(
    MODEL_NAME,
    num_parents=len(all_parent_codes),
    num_children=len(child_codes)
).to(device)

# --------------- TRAINER SETUP -------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    eval_strategy="epoch",
    logging_steps=25,
    save_total_limit=1,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
def compute_metrics(eval_pred):
    logits_parent, logits_child = eval_pred.predictions[0], eval_pred.predictions[1]
    labels_parent, labels_child = eval_pred.label_ids[0], eval_pred.label_ids[1]
    preds_parent = (torch.sigmoid(torch.tensor(logits_parent)).numpy() > THRESH_PARENT).astype(int)
    preds_child  = (torch.sigmoid(torch.tensor(logits_child)).numpy()  > THRESH_CHILD).astype(int)
    metrics = {
        "parent_micro_f1": f1_score(labels_parent, preds_parent, average="micro", zero_division=0),
        "child_micro_f1":  f1_score(labels_child, preds_child, average="micro", zero_division=0),
        "child_macro_f1":  f1_score(labels_child, preds_child, average="macro", zero_division=0),
    }
    return metrics
class HierDataCollator:
    def __call__(self, batch):
        result = {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}
        result["labels"] = (result["labels_parent"], result["labels_child"])
        return result

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=HierDataCollator(),
    compute_metrics=compute_metrics
)

# --------------- TRAIN & EVALUATE -------------------------
print("\nTraining Hierarchical ClinicalBERT...")
trainer.train()

print("\nEvaluating...")
pred_result = trainer.predict(test_ds)
logits_parent, logits_child = pred_result.predictions[0], pred_result.predictions[1]
yprob_parent = torch.sigmoid(torch.tensor(logits_parent)).numpy()
yprob_child  = torch.sigmoid(torch.tensor(logits_child)).numpy()
y_pred_parent = (yprob_parent > THRESH_PARENT).astype(int)
y_pred_child  = (yprob_child  > THRESH_CHILD).astype(int)

# --------------- METRICS & OUTPUTS ------------------------
# Parent label F1s
parent_f1s = [f1_score(Y_parent_test[:,i], y_pred_parent[:,i], zero_division=0) for i in range(len(all_parent_codes))]
child_f1s  = [f1_score(Y_child_test[:,i],  y_pred_child[:,i],  zero_division=0) for i in range(len(child_codes))]

# F1 Plots
plt.figure(figsize=(8,2.5))
sns.barplot(x=all_parent_codes, y=parent_f1s)
plt.title("Parent (Site) F1 Per Label")
plt.ylabel("F1")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/hier_parent_f1_per_label.png", dpi=200)
plt.close()

plt.figure(figsize=(10,3))
sns.barplot(x=child_codes, y=child_f1s)
plt.title("Child (ICD) F1 Per Label")
plt.ylabel("F1")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/hier_child_f1_per_label.png", dpi=200)
plt.close()

# Global metrics
metrics = {
    "parent_micro_f1": f1_score(Y_parent_test, y_pred_parent, average="micro", zero_division=0),
    "parent_macro_f1": f1_score(Y_parent_test, y_pred_parent, average="macro", zero_division=0),
    "child_micro_f1":  f1_score(Y_child_test,  y_pred_child,  average="micro", zero_division=0),
    "child_macro_f1":  f1_score(Y_child_test,  y_pred_child,  average="macro", zero_division=0),
    "child_weighted_f1": f1_score(Y_child_test, y_pred_child, average='weighted', zero_division=0),
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("results/hierarchical_metrics.csv", index=False)

# Full classification report
child_report = classification_report(Y_child_test, y_pred_child, target_names=child_codes, zero_division=0)
with open("results/hier_child_classification_report.txt","w") as f:
    f.write(child_report)
parent_report = classification_report(Y_parent_test, y_pred_parent, target_names=all_parent_codes, zero_division=0)
with open("results/hier_parent_classification_report.txt","w") as f:
    f.write(parent_report)

# Confusion matrices for children (if N_TOP <= 10)
if len(child_codes) <= 10:
    conf_mats = multilabel_confusion_matrix(Y_child_test, y_pred_child)
    fig, axs = plt.subplots(len(child_codes), 1, figsize=(5, 2*len(child_codes)), constrained_layout=True)
    if len(child_codes) == 1:
        axs = [axs]
    for i, code in enumerate(child_codes):
        ax = axs[i]
        cm = conf_mats[i]
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cbar=False,
                    xticklabels=["Not "+code, code], yticklabels=["Not "+code, code])
        ax.set_title(f"Child Confusion for {code}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.suptitle("Label-wise Confusion Matrices (Hierarchical, children)", y=1.02)
    plt.savefig("results/hier_child_conf_matrix.png", dpi=200)
    plt.close()

# Print summary
print("\nGlobal Summary:")
print(metrics_df.T)
print("\nPer-Parent Label F1:")
for p, f1 in zip(all_parent_codes, parent_f1s):
    print(f"{p}: {f1:.3f}")
print("\nPer-Child Label F1:")
for c, f1 in zip(child_codes, child_f1s):
    print(f"{c}: {f1:.3f}")

print("\nResults and plots saved in 'results/'.")
