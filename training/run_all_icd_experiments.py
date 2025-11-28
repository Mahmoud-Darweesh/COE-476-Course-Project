import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score, f1_score, classification_report, hamming_loss, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertPreTrainedModel, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

os.makedirs("results", exist_ok=True)

# ---------- Data loading -------------
REAL_PATH = "DataICD/surgery_reports_icd_multilabel.csv"
SYN_PATH = "DataICD/synthetic_gpt_icd_data.csv"
SYNFOCUS_PATH = "DataICD/synthetic_gpt_icd_data_focused.csv"

def get_dataframe(path):
    df = pd.read_csv(path)
    df['icd_codes'] = df['icd_codes'].apply(eval)
    return df

def get_parent(code):
    return code[:4] if len(code) > 4 else code[:3]

# ---------- Dataset for torch ----------
class ICDTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = np.array(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

# ---------- Hierarchical ClinicalBERT ----------
class HierClinicalBERT(BertPreTrainedModel):
    def __init__(self, config, parent_codes, child_codes, hier_loss_weight=0.7):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(0.2)
        self.parent_head = torch.nn.Linear(config.hidden_size, len(parent_codes))
        self.child_head = torch.nn.Linear(config.hidden_size, len(child_codes))
        self.hier_loss_weight = hier_loss_weight
        self.parent_codes = parent_codes
        self.child_codes = child_codes
        # compute mapping at init
        self.child2parent_idx = [self.parent_codes.index(get_parent(c)) for c in self.child_codes]
        self.init_weights()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels_parent=None, labels_child=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.dropout(outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0])
        logits_parent = self.parent_head(pooled_output)
        logits_child = self.child_head(pooled_output)
        out = {"logits_parent": logits_parent, "logits_child": logits_child}
        if labels_parent is not None and labels_child is not None:
            loss_parent = torch.nn.functional.binary_cross_entropy_with_logits(logits_parent, labels_parent)
            loss_child_all = torch.nn.functional.binary_cross_entropy_with_logits(logits_child, labels_child, reduction='none')
            # Create parent mask: for each child, find its parent, mask only if parent is predicted positive
            with torch.no_grad():
                parent_probs = torch.sigmoid(logits_parent) > 0.3
                parent_mask = torch.stack([parent_probs[:, idx] for idx in self.child2parent_idx], dim=1)
            weighted_loss_child = (parent_mask.float() * loss_child_all + (1 - parent_mask.float()) * 0.5 * loss_child_all).mean()
            total_loss = self.hier_loss_weight * loss_parent + (1 - self.hier_loss_weight) * weighted_loss_child
            out["loss"] = total_loss
        return out

# ---------- Plotting helpers ----------
def plot_loss(history, outname):
    plt.figure()
    plt.plot(history['train'], label='Train loss')
    if 'val' in history:
        plt.plot(history['val'], label='Val loss')
    plt.legend(); plt.title("Loss Curve"); plt.tight_layout()
    plt.savefig(outname); plt.close()

def plot_roc_pr(Y_true, Y_score, icd_vocab, prefix):
    for i, c in enumerate(icd_vocab):
        if np.sum(Y_true[:,i]) == 0:
            continue
        fpr, tpr, _ = roc_curve(Y_true[:,i], Y_score[:,i])
        p, r, _ = precision_recall_curve(Y_true[:,i], Y_score[:,i])
        auc_ = auc(fpr, tpr)
        ap_ = average_precision_score(Y_true[:,i], Y_score[:,i])
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={auc_:.2f}')
        plt.plot([0,1],[0,1],'--')
        plt.title(f'ROC {c}'); plt.legend(); plt.tight_layout()
        plt.savefig(f'{prefix}_roc_{c}.png'); plt.close()

        plt.figure(); plt.plot(r, p, label=f'AP={ap_:.2f}')
        plt.title(f'PR {c}'); plt.legend(); plt.tight_layout()
        plt.savefig(f'{prefix}_pr_{c}.png'); plt.close()

def barplot(scores, names, outname):
    plt.figure(figsize=(8,3))
    sns.barplot(x=names, y=scores)
    plt.xticks(rotation=45,ha='right')
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

# ---------- Multilabel stratified split helper ---------
def stratified_ml_split(X, Y, test_size=0.2, random_state=42):
    # Converts input to the right shape/type and splits
    Xarr = np.array(X).reshape(-1, 1)
    Yarr = np.array(Y)
    X_train, Y_train, X_test, Y_test = iterative_train_test_split(Xarr, Yarr, test_size)
    # flatten back to 1D for pandas compatibility
    return X_train.ravel(), X_test.ravel(), Y_train, Y_test

# ---------- Experiment runner ----------
def run_experiment(name, X_train, X_test, Y_train, Y_test, icd_vocab, code_counts, model_type, parent_codes=None, child_codes=None):
    print(f"\n======== Running: {name} ========")
    result_dir = f"results/{name.replace(' ', '_')}"
    os.makedirs(result_dir, exist_ok=True)

    # TF-IDF + LR
    if model_type == 'tfidf_lr':
        tfidf = TfidfVectorizer(max_features=10000)
        Xtr = tfidf.fit_transform(X_train)
        Xte = tfidf.transform(X_test)
        model = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=3, solver="liblinear"))
        model.fit(Xtr, Y_train)
        Y_scores = model.decision_function(Xte)
        Y_pred = (Y_scores > 0.2).astype(int)
        with open(f"{result_dir}/model.pkl","wb") as f: pickle.dump(model,f)
    elif model_type.startswith('bert'):
        if model_type == 'bert_base':
            model_name = "bert-base-uncased"
        elif model_type == 'bert_clinical':
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
        else:
            raise ValueError("Unknown BERT model_type! Got: ", model_type)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        train_ds = ICDTextDataset(X_train, Y_train, tokenizer)
        test_ds = ICDTextDataset(X_test, Y_test, tokenizer)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=Y_train.shape[1], problem_type="multi_label_classification").to('cuda' if torch.cuda.is_available() else 'cpu')
        training_args = TrainingArguments(
            output_dir=result_dir, num_train_epochs=4, per_device_train_batch_size=4,
            per_device_eval_batch_size=8, learning_rate=3e-5, eval_strategy="steps", load_best_model_at_end=True,)
        history = {'train':[], 'val':[]}
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs>0.2).astype(int)
            return {'micro_f1': f1_score(labels, preds, average='micro',zero_division=0),
                    'macro_f1': f1_score(labels, preds, average='macro',zero_division=0)}
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics)
        trainer.train()
        Y_scores, Y_pred = [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(torch.utils.data.DataLoader(test_ds, batch_size=8)):
                input_ids, att_mask = batch['input_ids'].to(model.device), batch['attention_mask'].to(model.device)
                logits = model(input_ids=input_ids, attention_mask=att_mask).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                Y_scores.append(probs); Y_pred.append((probs>0.2).astype(int))
        Y_scores = np.vstack(Y_scores)
        Y_pred = np.vstack(Y_pred)
    elif model_type == 'hier_clinbert':
        # parent/child setup
        tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        class HierICDDataset(Dataset):
            def __init__(self, texts, y_par, y_ch, tokenizer, max_len=256):
                self.texts = list(texts)
                self.y_parent = np.array(y_par, dtype=np.float32)
                self.y_child = np.array(y_ch, dtype=np.float32)
                self.tokenizer = tokenizer; self.max_len = max_len
            def __len__(self): return len(self.texts)
            def __getitem__(self, idx):
                enc = self.tokenizer(str(self.texts[idx]), truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt')
                item = {k: v.squeeze(0) for k, v in enc.items()}
                item["labels_parent"] = torch.tensor(self.y_parent[idx], dtype=torch.float32)
                item["labels_child"] = torch.tensor(self.y_child[idx], dtype=torch.float32)
                return item
        train_ds = HierICDDataset(X_train, Y_train[0], Y_train[1], tokenizer)
        test_ds = HierICDDataset(X_test, Y_test[0], Y_test[1], tokenizer)
        model = HierClinicalBERT.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", parent_codes=parent_codes, child_codes=child_codes)
        training_args = TrainingArguments(
            output_dir=result_dir, num_train_epochs=4, per_device_train_batch_size=4,
            per_device_eval_batch_size=8, learning_rate=3e-5, eval_strategy="steps", load_best_model_at_end=True)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds)
        trainer.train()
        model.eval()
        yscore_par, yscore_ch, ypred_par, ypred_ch = [],[],[],[]
        with torch.no_grad():
            for batch in tqdm(torch.utils.data.DataLoader(test_ds, batch_size=8)):
                inp, att = batch['input_ids'].to(model.device), batch['attention_mask'].to(model.device)
                logits = model(inp, attention_mask=att)
                probs_par = torch.sigmoid(logits['logits_parent']).cpu().numpy()
                probs_ch = torch.sigmoid(logits['logits_child']).cpu().numpy()
                yscore_par.append(probs_par); yscore_ch.append(probs_ch)
                ypred_par.append((probs_par>0.3).astype(int)); ypred_ch.append((probs_ch>0.2).astype(int))
        Y_scores = (np.vstack(yscore_par), np.vstack(yscore_ch))
        Y_pred = (np.vstack(ypred_par), np.vstack(ypred_ch))
        # Per-child/parent metrics as below
    else: raise ValueError("Unknown model type.")

    # --- Reporting ---
    if model_type != 'hier_clinbert':
        metrics = {
            'micro_f1': f1_score(Y_test, Y_pred, average='micro', zero_division=0),
            'macro_f1': f1_score(Y_test, Y_pred, average='macro', zero_division=0),
            'hamming_loss': hamming_loss(Y_test, Y_pred),
            'accuracy': accuracy_score(Y_test, Y_pred)
        }
        print(metrics)
        pd.DataFrame([metrics]).to_csv(f"{result_dir}/metrics.csv")
        report = classification_report(Y_test, Y_pred, target_names=icd_vocab, zero_division=0)
        with open(f"{result_dir}/report.txt","w") as f: f.write(report)
        barplot([f1_score(Y_test[:,i], Y_pred[:,i], zero_division=0) for i in range(len(icd_vocab))], icd_vocab, f"{result_dir}/f1_bar.png")
        plot_roc_pr(Y_test, Y_scores, icd_vocab, f"{result_dir}/prroc")
    else:
        # Hier: tuple of (parent, child) for both
        mp = lambda ytrue, ypred, avg: f1_score(ytrue, ypred, average=avg, zero_division=0)
        metrics = {
            'parent_micro_f1':mp(Y_test[0],Y_pred[0],'micro'),'parent_macro_f1':mp(Y_test[0],Y_pred[0],'macro'),
            'child_micro_f1':mp(Y_test[1],Y_pred[1],'micro'),'child_macro_f1':mp(Y_test[1],Y_pred[1],'macro'),
        }
        pd.DataFrame([metrics]).to_csv(f"{result_dir}/metrics.csv")
        with open(f"{result_dir}/child_report.txt","w") as f:
            f.write(classification_report(Y_test[1], Y_pred[1], target_names=child_codes, zero_division=0))
        with open(f"{result_dir}/parent_report.txt","w") as f:
            f.write(classification_report(Y_test[0], Y_pred[0], target_names=parent_codes, zero_division=0))
        barplot([f1_score(Y_test[1][:,i], Y_pred[1][:,i], zero_division=0) for i in range(len(child_codes))], child_codes, f"{result_dir}/f1_bar_child.png")
        barplot([f1_score(Y_test[0][:,i], Y_pred[0][:,i], zero_division=0) for i in range(len(parent_codes))], parent_codes, f"{result_dir}/f1_bar_parent.png")
        # No ROC/PR for parent/child (too many graphs), but can do it if desired

# -------------- All Experiments Loop ---------------
def experiment_loop():
    datasets = [
        ("real", lambda: [get_dataframe(REAL_PATH)]),
        ("real+synthetic+focused", lambda: [get_dataframe(REAL_PATH), get_dataframe(SYN_PATH), get_dataframe(SYNFOCUS_PATH)]),
        ("synthetic+focused", lambda: [get_dataframe(SYN_PATH), get_dataframe(SYNFOCUS_PATH)]),
    ]
    model_types = ['tfidf_lr','bert_base','bert_clinical','hier_clinbert']
    tops = [5,10]
    for topn in tops:
        for dname, dfunc in datasets:
            
            
            
            dfs = dfunc()
            df = pd.concat(dfs, ignore_index=True)
            all_codes = [c for codes in df['icd_codes'] for c in codes]
            code_counts = pd.Series(all_codes).value_counts()
            icd_vocab = code_counts.index[:topn].tolist()
            # Only keep rows with ICDs in vocab
            df['icd_codes_filtered'] = df['icd_codes'].apply(lambda codes: [c for c in codes if c in icd_vocab])
            df = df[df['icd_codes_filtered'].map(len)>0].reset_index(drop=True)
            # Always test on real!
            df_real = get_dataframe(REAL_PATH)
            df_real['icd_codes_filtered'] = df_real['icd_codes'].apply(lambda codes: [c for c in codes if c in icd_vocab])
            df_real = df_real[df_real['icd_codes_filtered'].map(len)>0].reset_index(drop=True)
            mlb = MultiLabelBinarizer(classes=icd_vocab)
            Y = mlb.fit_transform(df['icd_codes_filtered'])
            Y_real = mlb.transform(df_real['icd_codes_filtered'])
            X, X_real = df['report_text'], df_real['report_text']

            # Stratified split (on real)
            Xtr, Xte, Ytr, Yte = stratified_ml_split(X_real, Y_real, test_size=0.2)

            if dname=="real":
                trainX, trainY = Xtr, Ytr
            else:
                # Train on all synthetic + real train
                trainX = pd.concat([pd.Series(Xtr), pd.Series(X)], ignore_index=True)
                trainY = np.vstack([Ytr, Y])
            # Test only on Xte, Yte
      
            for mtype in model_types:
                
                # if topn == 5 and dname in ["real+synthetic+focused", "real"] and mtype in ['bert_base','bert_clinical']:
                #     continue  # skip heavy models for this combo
                # elif topn == 5 and dname == "real" and mtype == 'hier_clinbert':
                #     continue  # skip heavy models for this combo
                
                if mtype != 'tfidf_lr': continue # skip TFIDF_LR except for real
                expt_name = f"{mtype}__{dname}__top{topn}"
                if mtype == "hier_clinbert":
                    # Hierarchical setup
                    # CHILD
                    child_codes = icd_vocab
                    # PARENT
                    parent_codes = sorted(list({get_parent(c) for c in child_codes}))
                    parent_mlb = MultiLabelBinarizer(classes=parent_codes)
                    df['parent_codes'] = df['icd_codes_filtered'].apply(lambda xs: list({get_parent(c) for c in xs}))
                    df_real['parent_codes'] = df_real['icd_codes_filtered'].apply(lambda xs: list({get_parent(c) for c in xs}))
                    Y_parent = parent_mlb.fit_transform(df['parent_codes'])
                    Y_parent_real = parent_mlb.transform(df_real['parent_codes'])
                    # Let's create Train and Test DataFrames first
                    # train_df, test_df = train_test_split(df_real, test_size=0.2, random_state=197, stratify=mlb.transform(df_real["icd_codes_filtered"]))
                    # from skmultilearn.model_selection import iterative_train_test_split

                    X = df_real.reset_index(drop=True)
                    Y = mlb.transform(X["icd_codes_filtered"])
                    X_np = np.arange(len(X)).reshape(-1, 1)

                    # returns: X_train, Y_train, X_test, Y_test (X_train shape = (n_samples, 1), values are indices)
                    X_train_idx, Y_train, X_test_idx, Y_test = iterative_train_test_split(X_np, Y, test_size=0.2)
                    train_idx = X_train_idx.flatten()
                    test_idx = X_test_idx.flatten()
                    train_df = X.iloc[train_idx].reset_index(drop=True)
                    test_df = X.iloc[test_idx].reset_index(drop=True)
                    trainX = train_df["report_text"].values
                    Xte = test_df["report_text"].values

                    train_child = mlb.transform(train_df["icd_codes_filtered"])
                    test_child  = mlb.transform(test_df["icd_codes_filtered"])
                    train_parent = parent_mlb.transform(train_df["parent_codes"])
                    test_parent  = parent_mlb.transform(test_df["parent_codes"])

                    Ytr_hier = (train_parent, train_child)
                    Yte_hier = (test_parent, test_child)
                    run_experiment(
                        expt_name,
                        trainX, Xte,
                        Ytr_hier, Yte_hier,
                        child_codes, code_counts, 'hier_clinbert', parent_codes, child_codes
                    )
                    # Ytr_hier = (parent_mlb.transform(df_real['parent_codes'].iloc[:len(Xtr)]), Ytr)
                    # Yte_hier = (parent_mlb.transform(df_real['parent_codes'].iloc[len(Xtr):]), Yte)
                    # run_experiment(expt_name, trainX, Xte, Ytr_hier, Yte_hier, child_codes, code_counts, 'hier_clinbert', parent_codes, child_codes)
                else:
                    run_experiment(expt_name, trainX, Xte, trainY, Yte, icd_vocab, code_counts, mtype)

if __name__ == "__main__":
    experiment_loop()