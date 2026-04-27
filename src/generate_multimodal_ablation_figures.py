import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def apply_pub_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def model_order_from_data(df: pd.DataFrame) -> List[str]:
    preferred = ["voting", "et", "xgb", "rf", "lgbm"]
    present = [m for m in preferred if m in set(df["model"])]
    other = sorted(list(set(df["model"]) - set(preferred)))
    return present + other


def save_png_pdf(fig: plt.Figure, base_path_no_ext: str) -> None:
    fig.savefig(f"{base_path_no_ext}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base_path_no_ext}.pdf", bbox_inches="tight")


def make_heatmap(best_per_k: pd.DataFrame, outdir: str) -> None:
    endpoints = sorted(best_per_k["endpoint"].unique())
    order_models = model_order_from_data(best_per_k)
    k_order = [5, 4, 3, 2, 1]

    fig, axes = plt.subplots(1, len(endpoints), figsize=(7 * len(endpoints), 5), constrained_layout=True)
    if len(endpoints) == 1:
        axes = [axes]

    for ax, endpoint in zip(axes, endpoints):
        df_ep = best_per_k[best_per_k["endpoint"] == endpoint].copy()
        pivot = (
            df_ep.pivot(index="model", columns="n_modalities", values="mean_weighted_f1")
            .reindex(index=order_models, columns=k_order)
        )

        vals = pivot.to_numpy(dtype=float)
        data_min = float(np.nanmin(vals))
        data_max = float(np.nanmax(vals))
        span = max(data_max - data_min, 1e-6)
        pad = max(0.0005, 0.05 * span)
        vmin = max(0.0, data_min - pad)
        vmax = min(1.0, data_max + pad)

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="viridis",
            cbar=True,
            linewidths=0.5,
            linecolor="white",
            vmin=vmin,
            vmax=vmax,
            annot_kws={"fontsize": 9, "color": "black"},
            cbar_kws={"label": "Weighted F1"},
            ax=ax,
        )
        ax.set_title(f"Best Weighted F1 Heatmap ({endpoint})")
        ax.set_xlabel("Number of Modalities (k)")
        ax.set_ylabel("Model")

    save_png_pdf(fig, os.path.join(outdir, "modality_ablation_heatmap"))
    plt.close(fig)


def make_trend(best_per_k: pd.DataFrame, outdir: str) -> None:
    endpoints = sorted(best_per_k["endpoint"].unique())
    order_models = model_order_from_data(best_per_k)
    palette = sns.color_palette("colorblind", n_colors=max(5, len(order_models)))
    color_map = {m: palette[i] for i, m in enumerate(order_models)}

    fig, axes = plt.subplots(1, len(endpoints), figsize=(7 * len(endpoints), 5), constrained_layout=True)
    if len(endpoints) == 1:
        axes = [axes]

    for ax, endpoint in zip(axes, endpoints):
        df_ep = best_per_k[best_per_k["endpoint"] == endpoint].copy()
        for model in order_models:
            d = df_ep[df_ep["model"] == model].sort_values("n_modalities")
            if d.empty:
                continue
            x = d["n_modalities"].values
            y = d["mean_weighted_f1"].values
            yerr = d["std_weighted_f1"].fillna(0.0).values
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3,
                linewidth=2,
                label=model,
                color=color_map[model],
            )
        ax.set_title(f"Best-per-k Trend ({endpoint})")
        ax.set_xlabel("Number of Modalities (k)")
        ax.set_ylabel("Weighted F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.25)

    axes[0].legend(title="Model", frameon=True)
    save_png_pdf(fig, os.path.join(outdir, "modality_ablation_trend"))
    plt.close(fig)


def make_ranking_table(best_per_k: pd.DataFrame, outdir: str) -> pd.DataFrame:
    full_modal = (
        best_per_k[best_per_k["n_modalities"] == 5][["endpoint", "model", "mean_weighted_f1"]]
        .rename(columns={"mean_weighted_f1": "f1_k5"})
    )
    ranking = (
        best_per_k.groupby(["endpoint", "model"], as_index=False)
        .agg(avg_best_f1_over_k=("mean_weighted_f1", "mean"))
        .merge(full_modal, on=["endpoint", "model"], how="left")
        .sort_values(["endpoint", "avg_best_f1_over_k", "f1_k5"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    ranking["rank_within_endpoint"] = (
        ranking.groupby("endpoint")["avg_best_f1_over_k"].rank(ascending=False, method="min").astype(int)
    )
    ranking = ranking[["endpoint", "rank_within_endpoint", "model", "avg_best_f1_over_k", "f1_k5"]]
    ranking.to_csv(os.path.join(outdir, "modality_ablation_model_ranking.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, max(2.4, 0.36 * (len(ranking) + 2))))
    ax.axis("off")
    show_df = ranking.copy()
    show_df["avg_best_f1_over_k"] = show_df["avg_best_f1_over_k"].map(lambda x: f"{x:.3f}")
    show_df["f1_k5"] = show_df["f1_k5"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    table = ax.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)
    ax.set_title("Model Ranking Table (Compact)", pad=12)

    save_png_pdf(fig, os.path.join(outdir, "modality_ablation_ranking_table"))
    plt.close(fig)
    return ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-quality figures for multimodal ablation results.")
    parser.add_argument(
        "--best-per-k",
        default=os.path.join("results_ensemble", "modality_ablation_best_per_k.csv"),
        help="Path to modality_ablation_best_per_k.csv",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join("paper_figures", "modality_ablation"),
        help="Directory for saving publication-quality figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.best_per_k):
        raise FileNotFoundError(
            f"Input file not found: {args.best_per_k}. "
            "Run training/ensemble_survrec_multimodal_ablation.py first."
        )

    apply_pub_style()
    best_per_k = pd.read_csv(args.best_per_k)

    required = {"endpoint", "model", "n_modalities", "mean_weighted_f1", "std_weighted_f1"}
    missing = required - set(best_per_k.columns)
    if missing:
        raise ValueError(f"Missing required columns in best-per-k file: {sorted(missing)}")

    make_heatmap(best_per_k, args.outdir)
    make_trend(best_per_k, args.outdir)
    ranking = make_ranking_table(best_per_k, args.outdir)

    print("[done] Saved paper-quality figures and table in:")
    print(f"  - {args.outdir}")
    print(f"[done] Ranking rows: {len(ranking)}")


if __name__ == "__main__":
    main()
