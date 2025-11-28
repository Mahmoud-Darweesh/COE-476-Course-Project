import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
PAPERFIGS_DIR = "paper_figures"
os.makedirs(PAPERFIGS_DIR, exist_ok=True)

def parse_classification_report(path):
    """Extract per-label and overall metrics from a sklearn classification report text file."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Parse per-label lines: label p r f1 support
    matches = re.findall(r"^\s*([A-Za-z0-9\.\s]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9]+)", text, flags=re.MULTILINE)
    col_names = ["label", "precision", "recall", "f1", "support"]
    data = []
    for row in matches:
        label, precision, recall, f1, support = row
        if label.strip() in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
            continue  # Skip global lines for now
        data.append(dict(label=label.strip(), precision=float(precision), recall=float(recall), f1=float(f1), support=int(support)))
    df = pd.DataFrame(data)
    # Parse summary metrics
    sum_metrics = {}
    for avgtype in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
        p = re.search(r"^\s*"+re.escape(avgtype)+r"\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9]+)", text, flags=re.MULTILINE)
        if p:
            sum_metrics[avgtype] = dict(precision=float(p[1]), recall=float(p[2]), f1=float(p[3]), support=int(p[4]))
    return df, sum_metrics

def find_reports(results_dir):
    """Collect all report files and their experiment meta-info."""
    expts = []
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if fname.endswith("report.txt"):
                ftype = fname.replace(".txt","")
                expt = os.path.relpath(root, results_dir)
                expts.append((root, fname, expt, ftype))
    return expts

def aggregate_summary():
    # Parse all main report files and build a results summary dataframe
    data = []
    experiments = find_reports(RESULTS_DIR)
    for root, fname, expt, ftype in experiments:
        fpath = os.path.join(root, fname)
        # e.g. expt = bert_base__real__top5, ftype = report, child_report, parent_report, etc.
        try:
            df, sums = parse_classification_report(fpath)
            key = expt+"_"+ftype
            # Get global metrics
            macro_f1 = sums.get("macro avg",{}).get("f1", np.nan)
            micro_f1 = sums.get("micro avg",{}).get("f1", np.nan)
            weighted_f1 = sums.get("weighted avg",{}).get("f1", np.nan)
            macro_prec = sums.get("macro avg",{}).get("precision", np.nan)
            micro_prec = sums.get("micro avg",{}).get("precision", np.nan)
            macro_rec = sums.get("macro avg",{}).get("recall", np.nan)
            micro_rec = sums.get("micro avg",{}).get("recall", np.nan)
            data.append({
                "experiment": expt,
                "report_type": ftype,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "weighted_f1": weighted_f1,
                "macro_precision": macro_prec,
                "micro_precision": micro_prec,
                "macro_recall": macro_rec,
                "micro_recall": micro_rec,
            })
            df['experiment'] = expt
            df['report_type'] = ftype
            # Save per-label df (can be merged for per-label fig later)
            df.to_csv(f"{PAPERFIGS_DIR}/{expt}_{ftype}_perlabel.csv", index=False)
        except Exception as e:
            print(f"Error parsing {fpath}: {e}")
    df_sum = pd.DataFrame(data)
    df_sum.to_csv(f"{PAPERFIGS_DIR}/ALL_EXPERIMENT_SUMMARY.csv", index=False)
    return df_sum

def get_model_labels(experiment):
    # returns (model, datacombo, topk)
    # e.g. "bert_base__real__top10"
    parts = experiment.split("__")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return experiment, '', ''

# 1. BAR CHART: Macro/Micro/Weighted F1 for all models, all configs
def make_summary_f1_bar(df_sum):
    # Only use main reports (not parent/child for hierarchical, summary from child_report)
    plotdf = df_sum[
        (df_sum['report_type']=='report') | ((df_sum['report_type']=='child_report') & df_sum['experiment'].str.contains('hier_clinbert'))
    ].copy()
    # Parse experiment tags
    plotdf[["model","data","topN"]] = plotdf['experiment'].apply(lambda x: pd.Series(get_model_labels(x)))
    # Order
    order = ['tfidf_lr','bert_base','bert_clinical','hier_clinbert']
    plt.figure(figsize=(12,5))
    sns.barplot(
        data=plotdf, x="model", y="macro_f1", hue="data",
        order=order, errorbar=None
    )
    plt.title("Macro F1 Score by Model and Data Setting")
    plt.ylim(0,1)
    plt.savefig(f"{PAPERFIGS_DIR}/macroF1_model_data.png")
    plt.close()

    # Also bar for micro_f1
    plt.figure(figsize=(12,5))
    sns.barplot(
        data=plotdf, x="model", y="micro_f1", hue="data",
        order=order, errorbar=None
    )
    plt.title("Micro F1 Score by Model and Data Setting")
    plt.ylim(0,1)
    plt.savefig(f"{PAPERFIGS_DIR}/microF1_model_data.png")
    plt.close()

def plot_f1_perlabel_csv(csvfile, outname):
    # Plots per-label F1, with labels on x-axis
    df = pd.read_csv(csvfile)
    plt.figure(figsize=(10,4))
    sns.barplot(x="label", y="f1", data=df)
    plt.title(f"F1 per ICD code for {os.path.basename(csvfile)}")
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

def gather_and_plot_perlabel():
    # For best experiment/model (e.g. clinicalBERT_real+synthetic+focused_top10), plot per-label F1 bars
    for fname in os.listdir(PAPERFIGS_DIR):
        if fname.endswith("_perlabel.csv"):
            plot_f1_perlabel_csv(os.path.join(PAPERFIGS_DIR,fname),
                                 os.path.join(PAPERFIGS_DIR,fname.replace("_perlabel.csv", "_f1bar.png")))

def find_and_plot_curves():
    # Find and plot ROC and PR curves if they exist (saved by your modeling code earlier!)
    for root, dirs, files in os.walk(RESULTS_DIR):
        for fname in files:
            if fname.endswith("_roc.png") or fname.endswith("_pr.png") or fname.endswith("_cmatrix.png") or fname.endswith("_calib.png"):
                fpath = os.path.join(root, fname)
                dest = os.path.join(PAPERFIGS_DIR, os.path.relpath(fpath, RESULTS_DIR).replace("\\","_"))
                try:
                    plt.imread(fpath)
                    import shutil; shutil.copyfile(fpath, dest)
                    print(f"Copied curve fig to {dest}")
                except Exception as e:
                    print(f"Could not copy image {fpath}: {e}")

def main():
    df_sum = aggregate_summary()
    print("Summary metrics across all experiments saved as ALL_EXPERIMENT_SUMMARY.csv")
    make_summary_f1_bar(df_sum)
    gather_and_plot_perlabel()
    find_and_plot_curves()
    print(f"Figures saved to {PAPERFIGS_DIR}/. Tip: add top figures to paper!")

if __name__ == "__main__":
    main()