import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from ensemble_survrec_multimodal_ablation import (
    ALL_MODALITIES,
    build_base_models,
    build_summary_tables,
    concatenate_subset_blocks,
    generate_modality_subsets,
    load_modality_blocks,
    parse_models,
    safe_auc,
    set_seed,
    threshold_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable exhaustive multimodal ablation with checkpointing and progress bars."
    )
    parser.add_argument("--endpoint", choices=["surv", "rec", "both"], default="both")
    parser.add_argument("--models", default="voting,et,xgb,rf,lgbm")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--ablation", choices=["exhaustive"], default="exhaustive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="results_ensemble")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--fresh", action="store_true", help="Ignore previous checkpoint CSV and start from scratch.")
    parser.add_argument("--heartbeat-seconds", type=int, default=30, help="How often to write progress_state.json.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_task_key(endpoint: str, subset_key: str, fold: int, model: str) -> Tuple[str, str, int, str]:
    return endpoint, subset_key, int(fold), model


def load_completed_keys(fold_csv_path: str) -> Set[Tuple[str, str, int, str]]:
    if not os.path.exists(fold_csv_path):
        return set()
    df = pd.read_csv(fold_csv_path)
    required = {"endpoint", "subset_key", "fold", "model"}
    if not required.issubset(df.columns):
        return set()
    return {
        make_task_key(r.endpoint, r.subset_key, int(r.fold), r.model)
        for r in df[list(required)].itertuples(index=False)
    }


def append_rows(csv_path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def write_progress_state(state_path: str, state: Dict) -> None:
    state["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def evaluate_single_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    endpoint: str,
    subset_key: str,
    n_modalities: int,
    fold_idx: int,
    selected_models: Sequence[str],
    base_models: Dict[str, object],
) -> List[Dict]:
    rows: List[Dict] = []
    selected_base = [m for m in selected_models if m != "voting"]
    probas = {}

    # Fit base learners once per fold so voting can reuse probabilities.
    for model_name in selected_base:
        clf = clone(base_models[model_name])
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]
        probas[model_name] = proba

        thr, best_f1 = threshold_search(y_val, proba)
        pred = (proba >= thr).astype(int)
        rows.append(
            {
                "endpoint": endpoint,
                "model": model_name,
                "subset_key": subset_key,
                "subset_modalities": subset_key,
                "n_modalities": n_modalities,
                "fold": fold_idx,
                "threshold": thr,
                "weighted_f1": float(best_f1),
                "auc": safe_auc(y_val, proba),
                "acc": float(accuracy_score(y_val, pred)),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
            }
        )

    if "voting" in selected_models:
        vote_proba = np.column_stack([probas[m] for m in selected_base]).mean(axis=1)
        thr, best_f1 = threshold_search(y_val, vote_proba)
        pred = (vote_proba >= thr).astype(int)
        rows.append(
            {
                "endpoint": endpoint,
                "model": "voting",
                "subset_key": subset_key,
                "subset_modalities": subset_key,
                "n_modalities": n_modalities,
                "fold": fold_idx,
                "threshold": thr,
                "weighted_f1": float(best_f1),
                "auc": safe_auc(y_val, vote_proba),
                "acc": float(accuracy_score(y_val, pred)),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
            }
        )

    return rows


def dedupe_fold_csv(fold_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(fold_csv_path)
    key_cols = ["endpoint", "subset_key", "fold", "model"]
    df = df.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last")
    df.to_csv(fold_csv_path, index=False)
    return df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    fold_csv = os.path.join(args.outdir, "modality_ablation_fold_metrics.csv")
    summary_csv = os.path.join(args.outdir, "modality_ablation_summary.csv")
    best_csv = os.path.join(args.outdir, "modality_ablation_best_per_k.csv")
    ranking_csv = os.path.join(args.outdir, "modality_ablation_model_ranking.csv")
    state_json = os.path.join(args.outdir, "modality_ablation_progress_state.json")

    selected_models = parse_models(args.models)
    subsets = generate_modality_subsets(ALL_MODALITIES, args.ablation)
    endpoints = [args.endpoint] if args.endpoint != "both" else ["surv", "rec"]
    expected_subsets = 31
    if len(subsets) != expected_subsets:
        raise RuntimeError(f"Expected 31 modality subsets, got {len(subsets)}")

    total_tasks = len(endpoints) * len(subsets) * args.n_folds * len(selected_models)

    if args.fresh and os.path.exists(fold_csv):
        os.remove(fold_csv)

    completed = load_completed_keys(fold_csv) if args.resume and not args.fresh else set()
    completed = {k for k in completed if k[3] in set(selected_models) and k[0] in set(endpoints)}

    print(f"[info] Models: {selected_models}")
    print(f"[info] Endpoints: {endpoints}")
    print(f"[info] Subsets: {len(subsets)} (expected 31)")
    print(f"[info] Total fold-model tasks: {total_tasks}")
    print(f"[info] Already completed (resume): {len(completed)}")

    if args.dry_run:
        return

    pids, blocks, y_surv, y_rec = load_modality_blocks(ALL_MODALITIES)
    print(f"[info] Aligned patients: {len(pids)}")

    progress = tqdm(
        total=total_tasks,
        initial=min(len(completed), total_tasks),
        dynamic_ncols=True,
        smoothing=0.05,
        desc="Ablation Progress",
        unit="task",
    )

    last_heartbeat = datetime.utcnow()
    base_models = build_base_models(args.seed)

    for endpoint in endpoints:
        y_all = y_surv if endpoint == "surv" else y_rec
        valid = y_all != -1
        y = y_all[valid].astype(int)

        for subset in subsets:
            subset_key = "+".join(subset)
            X_subset = concatenate_subset_blocks(blocks, subset)[valid]
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

            for fold_idx, (tr, va) in enumerate(skf.split(X_subset, y)):
                need_for_fold = []
                for model in selected_models:
                    k = make_task_key(endpoint, subset_key, fold_idx, model)
                    if k not in completed:
                        need_for_fold.append(model)

                if not need_for_fold:
                    continue

                progress.set_postfix_str(
                    f"ep={endpoint} k={len(subset)} fold={fold_idx + 1}/{args.n_folds} missing={len(need_for_fold)}"
                )

                rows = evaluate_single_fold(
                    X_train=X_subset[tr],
                    y_train=y[tr],
                    X_val=X_subset[va],
                    y_val=y[va],
                    endpoint=endpoint,
                    subset_key=subset_key,
                    n_modalities=len(subset),
                    fold_idx=fold_idx,
                    selected_models=selected_models,
                    base_models=base_models,
                )

                rows_to_write = []
                for row in rows:
                    key = make_task_key(row["endpoint"], row["subset_key"], row["fold"], row["model"])
                    if key in completed:
                        continue
                    rows_to_write.append(row)
                    completed.add(key)

                append_rows(fold_csv, rows_to_write)
                progress.update(len(rows_to_write))

                now = datetime.utcnow()
                elapsed = (now - last_heartbeat).total_seconds()
                if elapsed >= max(1, args.heartbeat_seconds):
                    write_progress_state(
                        state_json,
                        {
                            "completed_tasks": len(completed),
                            "total_tasks": total_tasks,
                            "percent": round(100.0 * len(completed) / max(1, total_tasks), 3),
                            "endpoint": endpoint,
                            "subset_key": subset_key,
                            "subset_size": len(subset),
                            "fold": fold_idx,
                        },
                    )
                    last_heartbeat = now

    progress.close()

    if not os.path.exists(fold_csv):
        raise RuntimeError("No fold metrics were generated.")

    fold_df = dedupe_fold_csv(fold_csv)
    summary_df, best_df, ranking_df = build_summary_tables(fold_df)
    summary_df.to_csv(summary_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)

    write_progress_state(
        state_json,
        {
            "completed_tasks": len(completed),
            "total_tasks": total_tasks,
            "percent": round(100.0 * len(completed) / max(1, total_tasks), 3),
            "status": "done",
        },
    )

    print("[done] Resumable ablation finished.")
    print(f"  - {fold_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {best_csv}")
    print(f"  - {ranking_csv}")
    print(f"  - {state_json}")


if __name__ == "__main__":
    main()
