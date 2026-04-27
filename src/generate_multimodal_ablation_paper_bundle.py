import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_md_table(df: pd.DataFrame, out_path: str, title: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def save_png_pdf(fig: plt.Figure, base_path_no_ext: str) -> None:
    fig.savefig(f"{base_path_no_ext}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base_path_no_ext}.pdf", bbox_inches="tight")


def model_order_from_data(df: pd.DataFrame) -> List[str]:
    preferred = ["voting", "et", "xgb", "rf", "lgbm"]
    present = [m for m in preferred if m in set(df["model"])]
    rest = sorted(list(set(df["model"]) - set(preferred)))
    return present + rest


def build_main_table(best_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        best_df.pivot_table(
            index=["endpoint", "model"],
            columns="n_modalities",
            values="mean_weighted_f1",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={5: "F1_k5", 4: "F1_k4", 3: "F1_k3", 2: "F1_k2", 1: "F1_k1"})
    )
    best_overall = (
        best_df.sort_values(["endpoint", "model", "mean_weighted_f1"], ascending=[True, True, False])
        .groupby(["endpoint", "model"], as_index=False)
        .first()[["endpoint", "model", "n_modalities", "subset_key", "mean_weighted_f1"]]
        .rename(
            columns={
                "n_modalities": "best_k",
                "subset_key": "best_subset",
                "mean_weighted_f1": "best_f1",
            }
        )
    )
    main_table = pivot.merge(best_overall, on=["endpoint", "model"], how="left")
    return main_table.sort_values(["endpoint", "best_f1"], ascending=[True, False]).reset_index(drop=True)


def build_retention_table(best_df: pd.DataFrame) -> pd.DataFrame:
    k5 = best_df[best_df["n_modalities"] == 5][["endpoint", "model", "mean_weighted_f1"]].rename(
        columns={"mean_weighted_f1": "f1_k5"}
    )
    k1 = best_df[best_df["n_modalities"] == 1][["endpoint", "model", "mean_weighted_f1"]].rename(
        columns={"mean_weighted_f1": "f1_k1"}
    )
    out = k5.merge(k1, on=["endpoint", "model"], how="inner")
    out["abs_drop_k5_to_k1"] = out["f1_k5"] - out["f1_k1"]
    out["retention_pct"] = 100.0 * out["f1_k1"] / out["f1_k5"]
    return out.sort_values(["endpoint", "abs_drop_k5_to_k1"], ascending=[True, True]).reset_index(drop=True)


def build_clinical_gap_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    tmp = summary_df.copy()
    tmp["has_clinical"] = tmp["subset_key"].str.contains("clinical")
    out = (
        tmp.groupby(["endpoint", "has_clinical"], as_index=False)
        .agg(
            mean_weighted_f1=("mean_weighted_f1", "mean"),
            n_rows=("mean_weighted_f1", "size"),
        )
        .sort_values(["endpoint", "has_clinical"], ascending=[True, False])
    )
    return out.reset_index(drop=True)


def build_modality_frequency_table(best_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in best_df.iterrows():
        mods = str(r["subset_key"]).split("+")
        for m in mods:
            rows.append({"endpoint": r["endpoint"], "modality": m})
    freq = pd.DataFrame(rows).groupby(["endpoint", "modality"], as_index=False).size()
    freq = freq.rename(columns={"size": "count_in_best_subsets"})
    return freq.sort_values(["endpoint", "count_in_best_subsets"], ascending=[True, False]).reset_index(drop=True)


def plot_clinical_gap(clinical_gap_df: pd.DataFrame, outdir: str) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = clinical_gap_df.copy()
    d["has_clinical"] = d["has_clinical"].map({True: "With clinical", False: "Without clinical"})
    sns.barplot(
        data=d,
        x="endpoint",
        y="mean_weighted_f1",
        hue="has_clinical",
        palette="colorblind",
        ax=ax,
    )
    ax.set_title("Effect of Clinical Modality Presence")
    ax.set_xlabel("Endpoint")
    ax.set_ylabel("Mean Weighted F1")
    ax.set_ylim(0, 1)
    ax.legend(title="")
    save_png_pdf(fig, os.path.join(outdir, "modality_ablation_clinical_gap"))
    plt.close(fig)


def plot_modality_frequency(freq_df: pd.DataFrame, outdir: str) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(
        data=freq_df,
        x="modality",
        y="count_in_best_subsets",
        hue="endpoint",
        palette="colorblind",
        ax=ax,
    )
    ax.set_title("Modality Frequency in Best-Per-k Subsets")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Count")
    ax.legend(title="Endpoint")
    save_png_pdf(fig, os.path.join(outdir, "modality_ablation_modality_frequency"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tables/figures bundle for multimodal ablation paper section.")
    parser.add_argument("--summary", default=os.path.join("results_ensemble", "modality_ablation_summary.csv"))
    parser.add_argument("--best-per-k", default=os.path.join("results_ensemble", "modality_ablation_best_per_k.csv"))
    parser.add_argument("--outdir", default=os.path.join("paper_figures", "modality_ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    summary_df = pd.read_csv(args.summary)
    best_df = pd.read_csv(args.best_per_k)

    main_table = build_main_table(best_df)
    retention_table = build_retention_table(best_df)
    clinical_gap_table = build_clinical_gap_table(summary_df)
    modality_freq_table = build_modality_frequency_table(best_df)

    main_csv = os.path.join(args.outdir, "table_main_best_per_k.csv")
    retention_csv = os.path.join(args.outdir, "table_retention_k1_vs_k5.csv")
    clinical_gap_csv = os.path.join(args.outdir, "table_clinical_vs_nonclinical.csv")
    modality_freq_csv = os.path.join(args.outdir, "table_modality_frequency.csv")

    main_md = os.path.join(args.outdir, "table_main_best_per_k.md")
    retention_md = os.path.join(args.outdir, "table_retention_k1_vs_k5.md")
    clinical_gap_md = os.path.join(args.outdir, "table_clinical_vs_nonclinical.md")
    modality_freq_md = os.path.join(args.outdir, "table_modality_frequency.md")

    main_table.to_csv(main_csv, index=False)
    retention_table.to_csv(retention_csv, index=False)
    clinical_gap_table.to_csv(clinical_gap_csv, index=False)
    modality_freq_table.to_csv(modality_freq_csv, index=False)

    save_md_table(main_table, main_md, "Main Best-Per-k Table")
    save_md_table(retention_table, retention_md, "Retention Table: k=1 vs k=5")
    save_md_table(clinical_gap_table, clinical_gap_md, "Clinical vs Non-Clinical Table")
    save_md_table(modality_freq_table, modality_freq_md, "Modality Frequency in Best Subsets")

    plot_clinical_gap(clinical_gap_table, args.outdir)
    plot_modality_frequency(modality_freq_table, args.outdir)

    print("[done] Paper bundle generated:")
    for p in [
        main_csv,
        retention_csv,
        clinical_gap_csv,
        modality_freq_csv,
        main_md,
        retention_md,
        clinical_gap_md,
        modality_freq_md,
        os.path.join(args.outdir, "modality_ablation_clinical_gap.png"),
        os.path.join(args.outdir, "modality_ablation_modality_frequency.png"),
    ]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
