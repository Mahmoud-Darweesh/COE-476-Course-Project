import argparse
import itertools
import os
import random
import re
from typing import Dict, List, Sequence, Tuple

import h5py
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
DEFAULT_OUTDIR = "results_ensemble"
ALL_MODALITIES: Tuple[str, ...] = (
    "clinical",
    "semantic",
    "temporal",
    "pathological",
    "spatial",
)
EMBEDDING_FILES: Dict[str, str] = {
    "clinical": "Embeddings/clinical_embedding_512.h5",
    "semantic": "Embeddings/text_semantic_embeddings_512.h5",
    "temporal": "Embeddings/temporal_embedding_512.h5",
    "pathological": "Embeddings/pathological_embedding_512.h5",
    "spatial": "Embeddings/spatial_embeddings_512.h5",
}


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)


try:
    import torch

    torch.manual_seed(SEED)
except Exception:
    pass


def normalize_pid(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8").strip()
        except Exception:
            return str(x)
    s = str(x).strip()
    if s.isdigit():
        return str(int(s))
    m = re.search(r"\d+", s)
    if m:
        return str(int(m.group(0)))
    return s


def get_patient_emb_table(h5file: str) -> Dict[str, np.ndarray]:
    h5dict = {}
    with h5py.File(h5file, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                h5dict[name] = obj[()]

        f.visititems(visitor)

    pids = h5dict.get("patient_id", None)
    if pids is None:
        pids = h5dict.get("patient_ids", None)
    if pids is None:
        raise RuntimeError(f"patient_id not found in {h5file}")

    pids = [normalize_pid(x) for x in pids]
    emb = None
    for v in h5dict.values():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == len(pids) and v.shape[1] == 512:
            emb = v
            break
    if emb is None:
        raise RuntimeError(f"No [n,512] embedding matrix found in {h5file}")

    return dict(zip(pids, emb))


def load_targets(clinical_h5: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    with h5py.File(clinical_h5, "r") as f:
        pids = [normalize_pid(x) for x in f["patient_id"][:]]
        y_surv = np.array(f["surv_5yr_label"])
        y_rec = np.array(f["rec_2yr_label"])
    return pids, y_surv, y_rec


def load_modality_blocks(modalities: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    pid_set = None
    pid2emb_by_mod = {}

    for mod in modalities:
        table = get_patient_emb_table(EMBEDDING_FILES[mod])
        pid2emb_by_mod[mod] = table
        mod_pids = set(table.keys())
        pid_set = mod_pids if pid_set is None else (pid_set & mod_pids)

    if not pid_set:
        raise RuntimeError("No aligned patients found across selected modalities.")

    pids_sorted = np.array(sorted(pid_set))
    blocks = {mod: np.array([pid2emb_by_mod[mod][pid] for pid in pids_sorted]) for mod in modalities}

    target_pids, y_surv_raw, y_rec_raw = load_targets(EMBEDDING_FILES["clinical"])
    pid_to_idx = {pid: i for i, pid in enumerate(target_pids)}
    aligned_idx = [pid_to_idx[pid] for pid in pids_sorted]
    y_surv = y_surv_raw[aligned_idx]
    y_rec = y_rec_raw[aligned_idx]

    return pids_sorted, blocks, y_surv, y_rec


def concatenate_subset_blocks(blocks: Dict[str, np.ndarray], subset: Sequence[str]) -> np.ndarray:
    return np.concatenate([blocks[mod] for mod in subset], axis=1)


def threshold_search(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        preds = y_proba >= thr
        f1 = f1_score(y_true, preds, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def build_base_models(seed: int) -> Dict[str, object]:
    return {
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=350,
            max_depth=8,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.09,
            num_leaves=31,
            n_jobs=-1,
            random_state=seed,
            verbose=-1
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.07,
            eval_metric="logloss",
            random_state=seed,
        ),
    }


def generate_modality_subsets(modalities: Sequence[str], ablation: str) -> List[Tuple[str, ...]]:
    if ablation != "exhaustive":
        raise ValueError(f"Unsupported ablation mode: {ablation}")

    subsets = []
    for k in range(len(modalities), 0, -1):
        subsets.extend(itertools.combinations(modalities, k))
    subsets = sorted(subsets, key=lambda x: (-len(x), x))
    return subsets


def parse_models(models_csv: str) -> List[str]:
    raw = [m.strip().lower() for m in models_csv.split(",") if m.strip()]
    allowed = {"voting", "rf", "et", "lgbm", "xgb"}
    invalid = [m for m in raw if m not in allowed]
    if invalid:
        raise ValueError(f"Unsupported models requested: {invalid}. Allowed: {sorted(allowed)}")
    if not raw:
        raise ValueError("No models selected. Provide at least one model.")
    if "voting" in raw and len([m for m in raw if m != "voting"]) == 0:
        raise ValueError("Voting requires at least one base model (rf/et/lgbm/xgb).")
    return raw


def evaluate_subset_endpoint(
    X: np.ndarray,
    y: np.ndarray,
    endpoint: str,
    subset: Sequence[str],
    selected_models: Sequence[str],
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    valid_mask = y != -1
    Xv = X[valid_mask]
    yv = y[valid_mask].astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    base_models = build_base_models(seed)
    selected_base = [m for m in selected_models if m != "voting"]

    rows = []
    subset_key = "+".join(subset)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(Xv, yv)):
        Xtr, Xva = Xv[train_idx], Xv[val_idx]
        ytr, yva = yv[train_idx], yv[val_idx]
        fold_probas = {}

        for model_name in selected_base:
            clf = clone(base_models[model_name])
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xva)[:, 1]
            fold_probas[model_name] = proba

            thr, best_f1 = threshold_search(yva, proba)
            preds = (proba >= thr).astype(int)

            rows.append(
                {
                    "endpoint": endpoint,
                    "model": model_name,
                    "subset_key": subset_key,
                    "subset_modalities": subset_key,
                    "n_modalities": len(subset),
                    "fold": fold_idx,
                    "threshold": thr,
                    "weighted_f1": float(best_f1),
                    "auc": safe_auc(yva, proba),
                    "acc": float(accuracy_score(yva, preds)),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                }
            )

        if "voting" in selected_models:
            vote_components = selected_base
            vote_matrix = np.column_stack([fold_probas[m] for m in vote_components])
            vote_proba = vote_matrix.mean(axis=1)
            thr, best_f1 = threshold_search(yva, vote_proba)
            preds = (vote_proba >= thr).astype(int)

            rows.append(
                {
                    "endpoint": endpoint,
                    "model": "voting",
                    "subset_key": subset_key,
                    "subset_modalities": subset_key,
                    "n_modalities": len(subset),
                    "fold": fold_idx,
                    "threshold": thr,
                    "weighted_f1": float(best_f1),
                    "auc": safe_auc(yva, vote_proba),
                    "acc": float(accuracy_score(yva, preds)),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                }
            )

    return pd.DataFrame(rows)


def build_summary_tables(fold_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        fold_df.groupby(["endpoint", "model", "subset_key", "n_modalities"], as_index=False)
        .agg(
            mean_weighted_f1=("weighted_f1", "mean"),
            std_weighted_f1=("weighted_f1", "std"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_acc=("acc", "mean"),
            std_acc=("acc", "std"),
            mean_threshold=("threshold", "mean"),
            n_folds=("fold", "count"),
        )
        .sort_values(
            ["endpoint", "model", "n_modalities", "mean_weighted_f1"],
            ascending=[True, True, False, False],
        )
    )

    best_idx = summary.groupby(["endpoint", "model", "n_modalities"])["mean_weighted_f1"].idxmax()
    best_per_k = (
        summary.loc[best_idx]
        .sort_values(["endpoint", "model", "n_modalities"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    full_modal = best_per_k[best_per_k["n_modalities"] == 5][["endpoint", "model", "mean_weighted_f1"]].rename(
        columns={"mean_weighted_f1": "f1_k5"}
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
    return summary, best_per_k, ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exhaustive multimodal embedding-count ablation runner.")
    parser.add_argument("--endpoint", choices=["surv", "rec", "both"], default="both")
    parser.add_argument("--models", default="voting,et,xgb,rf,lgbm", help="Comma-separated: voting,et,xgb,rf,lgbm")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--ablation", choices=["exhaustive"], default="exhaustive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--dry-run", action="store_true", help="Print workload only; skip training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    selected_models = parse_models(args.models)
    subsets = generate_modality_subsets(ALL_MODALITIES, args.ablation)
    endpoint_targets = [args.endpoint] if args.endpoint != "both" else ["surv", "rec"]

    expected_subsets = sum(len(list(itertools.combinations(ALL_MODALITIES, k))) for k in range(1, 6))
    if len(subsets) != expected_subsets:
        raise RuntimeError(f"Subset generation mismatch: got {len(subsets)}, expected {expected_subsets}.")

    runs_per_endpoint = len(subsets) * len(selected_models)
    fold_rows_per_endpoint = runs_per_endpoint * args.n_folds

    print(f"[info] Models: {selected_models}")
    print(f"[info] Endpoints: {endpoint_targets}")
    print(f"[info] Subsets generated: {len(subsets)} (expected 31)")
    print(f"[info] Expected runs per endpoint: {runs_per_endpoint}")
    print(f"[info] Expected fold rows per endpoint: {fold_rows_per_endpoint}")

    if args.dry_run:
        print("[dry-run] Done.")
        return

    print("[info] Loading aligned modality blocks...")
    pids, blocks, y_surv, y_rec = load_modality_blocks(ALL_MODALITIES)
    print(f"[info] Aligned patients: {len(pids)}")

    all_fold_parts = []

    for endpoint in endpoint_targets:
        y = y_surv if endpoint == "surv" else y_rec
        print(f"\n[info] Endpoint: {endpoint}")

        for idx, subset in enumerate(subsets, start=1):
            subset_key = "+".join(subset)
            print(f"  [{idx:02d}/{len(subsets)}] subset={subset_key}")
            X_subset = concatenate_subset_blocks(blocks, subset)
            fold_df = evaluate_subset_endpoint(
                X=X_subset,
                y=y,
                endpoint=endpoint,
                subset=subset,
                selected_models=selected_models,
                n_folds=args.n_folds,
                seed=args.seed,
            )
            all_fold_parts.append(fold_df)

    fold_metrics = pd.concat(all_fold_parts, ignore_index=True)
    summary, best_per_k, ranking = build_summary_tables(fold_metrics)

    fold_path = os.path.join(args.outdir, "modality_ablation_fold_metrics.csv")
    summary_path = os.path.join(args.outdir, "modality_ablation_summary.csv")
    best_path = os.path.join(args.outdir, "modality_ablation_best_per_k.csv")
    ranking_path = os.path.join(args.outdir, "modality_ablation_model_ranking.csv")

    fold_metrics.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    best_per_k.to_csv(best_path, index=False)
    ranking.to_csv(ranking_path, index=False)

    print("\n[done] Saved:")
    print(f"  - {fold_path}")
    print(f"  - {summary_path}")
    print(f"  - {best_path}")
    print(f"  - {ranking_path}")
    print("[next] Generate paper figures with src/generate_multimodal_ablation_figures.py")


if __name__ == "__main__":
    main()
