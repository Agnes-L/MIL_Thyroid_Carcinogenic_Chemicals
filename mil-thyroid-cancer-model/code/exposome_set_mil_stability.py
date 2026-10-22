"""
Repeat-level stable ranking for the Exposure Set-MIL model, v9_level12.

Compared with v9_level12, this version keeps the same MIL design but uses explicit NTS Level 1/2 input file names:

  - One short seed per repeated 5-fold cross-validation run: 10, 20, 30, ...
  - No inner validation split and no inner validation seed.
  - Each repeat produces one complete out-of-fold (OOF) chemical ranking by
    merging the five held-out folds, so the ranking unit is a full 5-fold CV
    repeat rather than an individual fold.
  - Fold models are trained for a pre-specified fixed number of epochs.
  - The primary ranking criterion is cancer-direction OOF contribution conditional
    on detection:
        OOF_CancerDirection_MeanDelta_WhenDetected
        = mean(max(delta_logit, 0) among serum samples where the chemical is detected)
    Negative deltas contribute 0 to the numerator but remain in the detected-sample
    denominator. No additional detection-rate, target-count, or composite weighting
    is applied after attribution.
  - Signed contribution and control-direction contribution are retained as
    auxiliary directionality diagnostics.
  - The global Detection_Rate_File channel and NTS_level12_hazard_binary_20261009.csv target-activity file are used.

The chemical universe is the current NTS Level 1/2 batch:
  - 561 chemicals aligned across sample detections, detection frequency, and target-activity files.
  - Endogenous/exogenous labels are kept in chemical_metadata_level12_20261009.csv, but are not used as model features.
  - No low-frequency filtering and no Fisher/MWU input features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from exposome_set_mil_pipeline_when_detected_20261009 import (
    build_torch_classes,
    explain_counterfactual_exact,
    load_exposome_data,
    make_numeric_tokens,
    set_global_seed,
    torch_predict_logits,
)


NTS_LEVEL12_POSITIVE_FILE = "NTS_level12_positive_sample_detections_20261009.csv"
NTS_LEVEL12_NEGATIVE_FILE = "NTS_level12_negative_sample_detections_20261009.csv"
NTS_LEVEL12_FREQUENCY_FILE = "NTS_level12_detection_frequency_20261009.csv"
NTS_LEVEL12_HAZARD_FILE = "NTS_level12_hazard_binary_20261009.csv"

PRIMARY_RANKING_SCORE = "OOF_CancerDirection_MeanDelta_WhenDetected"
PRIMARY_RANK_COLUMN = "OOF_CancerDirection_MeanDelta_WhenDetected_Rank"
PRIMARY_PERCENTILE_COLUMN = "OOF_CancerDirection_MeanDelta_WhenDetected_PercentileRank"
LEGACY_POPULATION_RANK_COLUMN = "OOF_CancerDirection_MeanDelta_Rank"
TARGET_SUPPORTED_RANK_COLUMN = "Target_Supported_CancerDirection_WhenDetected_Rank"


def load_exposome_data_v9_level12(data_dir: Path):
    """Load current NTS Level 1/2 data with explicit v9 input file names."""
    loader_globals = load_exposome_data.__globals__
    overrides = {
        "POSITIVE_FILE": NTS_LEVEL12_POSITIVE_FILE,
        "NEGATIVE_FILE": NTS_LEVEL12_NEGATIVE_FILE,
        "FREQUENCY_FILE": NTS_LEVEL12_FREQUENCY_FILE,
        "HAZARD_FILE": NTS_LEVEL12_HAZARD_FILE,
    }
    previous_files = {key: loader_globals.get(key) for key in overrides}
    loader_globals.update(overrides)
    try:
        data = load_exposome_data(data_dir)
    finally:
        for key, previous in previous_files.items():
            if previous is None:
                loader_globals.pop(key, None)
            else:
                loader_globals[key] = previous
    data.summary.update(
        {
            "positive_file": NTS_LEVEL12_POSITIVE_FILE,
            "negative_file": NTS_LEVEL12_NEGATIVE_FILE,
            "frequency_file": NTS_LEVEL12_FREQUENCY_FILE,
            "hazard_file": NTS_LEVEL12_HAZARD_FILE,
            "input_version": "v9_level12_20261009",
        }
    )
    return data


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
    }
    out["auc"] = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan
    return out


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["accuracy", "balanced_accuracy", "f1", "auc", "final_train_loss_mean"]:
        if col not in metrics.columns:
            continue
        values = pd.to_numeric(metrics[col], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": col,
                "n_models": int(values.shape[0]),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
                "min": float(values.min()),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def build_metadata(data) -> pd.DataFrame:
    p = data.presence
    y = data.y
    cancer_mask = y == 1
    control_mask = y == 0
    meta = pd.DataFrame(
        {
            "chemical_index": np.arange(len(data.chemicals)),
            "SMILES": data.chemicals,
            "CAS": data.cas,
            "Detection_Rate_File": data.detection_rate,
            "Detected_Count_In_100": p.sum(axis=0).astype(int),
            "Cohort_Detection_Rate_FromPresence": p.mean(axis=0),
            "Cancer_Detected_Count": p[cancer_mask].sum(axis=0).astype(int),
            "Control_Detected_Count": p[control_mask].sum(axis=0).astype(int),
            "Cancer_Detection_Rate": p[cancer_mask].mean(axis=0),
            "Control_Detection_Rate": p[control_mask].mean(axis=0),
        }
    )
    meta["Detection_Rate_Difference_CancerMinusControl"] = (
        meta["Cancer_Detection_Rate"] - meta["Control_Detection_Rate"]
    )
    for k, name in enumerate(data.target_names):
        meta[name] = data.hazards[:, k]
    meta["Hazard_Target_Count"] = data.hazards.sum(axis=1)
    return meta


def train_torch_model_fixed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    id_embed_dim: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    device: str,
) -> Tuple[object, pd.DataFrame, Dict[str, float]]:
    """Train GatedAttentionMIL for a fixed number of epochs with no inner validation split."""
    torch, nn, DataLoader, TensorDataset, GatedAttentionMIL = build_torch_classes()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = GatedAttentionMIL(
        num_chemicals=x_train.shape[1],
        numeric_dim=x_train.shape[2],
        id_embed_dim=id_embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()
            train_losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "lr": float(optim.param_groups[0]["lr"]),
            }
        )

    history_df = pd.DataFrame(history)
    return (
        model,
        history_df,
        {
            "epochs": int(epochs),
            "final_train_loss": float(history_df["train_loss"].iloc[-1]),
        },
    )


def optional_threshold(value: Optional[float]) -> Optional[float]:
    if value is None or value < 0:
        return None
    return float(value)


def threshold_text(value: Optional[float]) -> str:
    if value is None:
        return "all"
    return f"{value:.2f}".replace(".", "p")


def ranking_filter_label(min_auc: Optional[float], min_ba: Optional[float]) -> str:
    parts = []
    if min_auc is not None:
        parts.append(f"auc_ge_{threshold_text(min_auc)}")
    if min_ba is not None:
        parts.append(f"ba_ge_{threshold_text(min_ba)}")
    return "_".join(parts) if parts else "all_repeats"


def build_ranking_filter_specs(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], str]:
    primary_min_auc = optional_threshold(args.ranking_min_auc)
    primary_min_ba = optional_threshold(args.ranking_min_balanced_accuracy)
    base_specs = [
        {"label": "all_repeats", "min_auc": None, "min_balanced_accuracy": None},
        {"label": "auc_ge_0p85", "min_auc": 0.85, "min_balanced_accuracy": None},
        {"label": "auc_ge_0p90", "min_auc": 0.90, "min_balanced_accuracy": None},
        {"label": "auc_ge_0p95", "min_auc": 0.95, "min_balanced_accuracy": None},
        {
            "label": "auc_ge_0p90_ba_ge_0p80",
            "min_auc": 0.90,
            "min_balanced_accuracy": 0.80,
        },
    ]
    primary_label = ranking_filter_label(primary_min_auc, primary_min_ba)
    if not any(spec["label"] == primary_label for spec in base_specs):
        base_specs.append(
            {
                "label": primary_label,
                "min_auc": primary_min_auc,
                "min_balanced_accuracy": primary_min_ba,
            }
        )
    return base_specs, primary_label


def repeat_passes_ranking_filter(repeat_row: Dict[str, float], spec: Dict[str, object]) -> bool:
    min_auc = spec["min_auc"]
    min_ba = spec["min_balanced_accuracy"]
    auc = repeat_row.get("auc", np.nan)
    ba = repeat_row.get("balanced_accuracy", np.nan)
    if min_auc is not None and (not np.isfinite(auc) or auc < float(min_auc)):
        return False
    if min_ba is not None and (not np.isfinite(ba) or ba < float(min_ba)):
        return False
    return True


def build_repeat_seeds(args: argparse.Namespace) -> List[int]:
    if args.seed_list:
        seeds = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]
        if len(seeds) != args.repeats:
            raise ValueError("--seed-list length must equal --repeats.")
        return seeds
    return [args.seed_start + i * args.seed_step for i in range(args.repeats)]


def validate_counterfactual_exact(
    model,
    x_batch: np.ndarray,
    *,
    batch_size: int,
    device: str,
    classifier_chunk: int,
    max_samples: int = 8,
) -> pd.DataFrame:
    """Small numerical check for fast deletion attribution against direct deletion."""
    if x_batch.shape[0] == 0:
        return pd.DataFrame()

    x_check = x_batch[: min(max_samples, x_batch.shape[0])].copy()
    base_fast, delta_fast, _ = explain_counterfactual_exact(
        model,
        x_check,
        batch_size=batch_size,
        device=device,
        classifier_chunk=classifier_chunk,
    )
    base_direct = torch_predict_logits(model, x_check, batch_size=batch_size, device=device)
    rows = [
        {
            "check": "base_logit_fast_vs_direct",
            "max_abs_error": float(np.max(np.abs(base_fast - base_direct))),
            "passed": bool(np.max(np.abs(base_fast - base_direct)) < 1e-5),
        }
    ]

    absent_mask = x_check[:, :, 0] <= 0
    absent_delta = np.abs(delta_fast[absent_mask])
    absent_max = float(absent_delta.max()) if absent_delta.size else 0.0
    rows.append(
        {
            "check": "absent_chemical_delta_zero",
            "max_abs_error": absent_max,
            "passed": bool(absent_max < 1e-8),
        }
    )

    detected_counts = (x_check[:, :, 0] > 0).sum(axis=1)
    normal_candidates = np.where(detected_counts >= 2)[0]
    if normal_candidates.size:
        sample_idx = int(normal_candidates[0])
        chem_idx = int(np.where(x_check[sample_idx, :, 0] > 0)[0][0])
        x_deleted = x_check[[sample_idx]].copy()
        x_deleted[0, chem_idx, :] = 0.0
        deleted_direct = torch_predict_logits(model, x_deleted, batch_size=1, device=device)[0]
        direct_delta = float(base_direct[sample_idx] - deleted_direct)
        fast_delta = float(delta_fast[sample_idx, chem_idx])
        rows.append(
            {
                "check": "detected_chemical_fast_delta_vs_direct_delete_nonempty_bag",
                "sample_index_in_validation_batch": sample_idx,
                "chemical_index": chem_idx,
                "fast_delta": fast_delta,
                "direct_delta": direct_delta,
                "max_abs_error": abs(fast_delta - direct_delta),
                "direction_check": "base_minus_deleted_logit",
                "passed": bool(abs(fast_delta - direct_delta) < 1e-5),
            }
        )

    single_candidates = np.where(detected_counts >= 1)[0]
    if single_candidates.size:
        source_idx = int(single_candidates[0])
        chem_idx = int(np.where(x_check[source_idx, :, 0] > 0)[0][0])
        x_single = np.zeros_like(x_check[[source_idx]])
        x_single[0, chem_idx, :] = x_check[source_idx, chem_idx, :]
        single_base, single_delta, _ = explain_counterfactual_exact(
            model,
            x_single,
            batch_size=1,
            device=device,
            classifier_chunk=classifier_chunk,
        )
        x_empty = x_single.copy()
        x_empty[0, chem_idx, :] = 0.0
        empty_direct = torch_predict_logits(model, x_empty, batch_size=1, device=device)[0]
        direct_delta = float(single_base[0] - empty_direct)
        fast_delta = float(single_delta[0, chem_idx])
        rows.append(
            {
                "check": "single_detected_chemical_delete_to_empty_bag",
                "sample_index_in_validation_batch": source_idx,
                "chemical_index": chem_idx,
                "fast_delta": fast_delta,
                "direct_delta": direct_delta,
                "max_abs_error": abs(fast_delta - direct_delta),
                "empty_bag_direct_logit_is_finite": bool(np.isfinite(empty_direct)),
                "passed": bool(np.isfinite(empty_direct) and abs(fast_delta - direct_delta) < 1e-5),
            }
        )

    return pd.DataFrame(rows)


def compute_repeat_stats(data, repeat_delta: np.ndarray, repeat_contribution: np.ndarray) -> Dict[str, np.ndarray]:
    p = data.presence.astype(np.float64)
    y = data.y.astype(int)
    n_samples = p.shape[0]
    detected_count = p.sum(axis=0)
    denom_detected = np.maximum(detected_count, 1.0)

    signed_sum = repeat_contribution.sum(axis=0)
    abs_sum = np.abs(repeat_contribution).sum(axis=0)
    cancer_sum = np.maximum(repeat_contribution, 0.0).sum(axis=0)
    control_sum = np.maximum(-repeat_contribution, 0.0).sum(axis=0)
    positive_count = ((repeat_delta > 0) & (p > 0)).sum(axis=0)

    out = {
        "OOF_Population_MeanDelta": signed_sum / n_samples,
        "OOF_Signed_MeanDelta": signed_sum / n_samples,
        "OOF_AbsPopulation_MeanDelta": abs_sum / n_samples,
        "OOF_CancerDirection_MeanDelta": cancer_sum / n_samples,
        "OOF_ControlDirection_MeanDelta": control_sum / n_samples,
        "OOF_Conditional_MeanDelta_WhenDetected": signed_sum / denom_detected,
        "OOF_CancerDirection_MeanDelta_WhenDetected": cancer_sum / denom_detected,
        "OOF_ControlDirection_MeanDelta_WhenDetected": control_sum / denom_detected,
        "OOF_Positive_Delta_Frequency_WhenDetected": positive_count / denom_detected,
        "OOF_Detected_Explanation_Count": detected_count,
    }

    group_specs = [
        ("CancerGroup", y == 1),
        ("ControlGroup", y == 0),
    ]
    for label, sample_mask in group_specs:
        group_p = p[sample_mask]
        group_delta = repeat_delta[sample_mask]
        group_contribution = repeat_contribution[sample_mask]
        group_detected = group_p.sum(axis=0)
        group_denom = np.maximum(group_detected, 1.0)
        group_positive_sum = np.maximum(group_contribution, 0.0).sum(axis=0)
        group_negative_sum = np.maximum(-group_contribution, 0.0).sum(axis=0)
        group_positive_count = ((group_delta > 0) & (group_p > 0)).sum(axis=0)

        out[f"OOF_{label}_PositiveMeanDelta_WhenDetected"] = group_positive_sum / group_denom
        out[f"OOF_{label}_NegativeMeanDelta_WhenDetected"] = group_negative_sum / group_denom
        out[f"OOF_{label}_PositiveDelta_Frequency_WhenDetected"] = group_positive_count / group_denom
        out[f"OOF_{label}_Detected_Explanation_Count"] = group_detected

    return out


def init_ranking_aggregator(n_chemicals: int, topk_values: List[int]) -> Dict[str, object]:
    keys = [
        "OOF_Population_MeanDelta",
        "OOF_Signed_MeanDelta",
        "OOF_AbsPopulation_MeanDelta",
        "OOF_CancerDirection_MeanDelta",
        "OOF_ControlDirection_MeanDelta",
        "OOF_Conditional_MeanDelta_WhenDetected",
        "OOF_CancerDirection_MeanDelta_WhenDetected",
        "OOF_ControlDirection_MeanDelta_WhenDetected",
        "OOF_Positive_Delta_Frequency_WhenDetected",
        "OOF_Detected_Explanation_Count",
        "OOF_CancerGroup_PositiveMeanDelta_WhenDetected",
        "OOF_CancerGroup_NegativeMeanDelta_WhenDetected",
        "OOF_CancerGroup_PositiveDelta_Frequency_WhenDetected",
        "OOF_CancerGroup_Detected_Explanation_Count",
        "OOF_ControlGroup_PositiveMeanDelta_WhenDetected",
        "OOF_ControlGroup_NegativeMeanDelta_WhenDetected",
        "OOF_ControlGroup_PositiveDelta_Frequency_WhenDetected",
        "OOF_ControlGroup_Detected_Explanation_Count",
    ]
    out = {key: np.zeros(n_chemicals, dtype=np.float64) for key in keys}
    out["top_counts"] = {k: np.zeros(n_chemicals, dtype=np.int64) for k in topk_values}
    out["eligible_repeat_count"] = 0
    out["top_rows"] = []
    return out


def update_ranking_aggregator(
    aggregator: Dict[str, object],
    *,
    data,
    repeat_stats: Dict[str, np.ndarray],
    topk_values: List[int],
    filter_label: str,
    repeat: int,
    repeat_seed: int,
) -> None:
    for key, value in repeat_stats.items():
        aggregator[key] += value
    aggregator["eligible_repeat_count"] += 1

    repeat_score = repeat_stats[PRIMARY_RANKING_SCORE]
    for k in topk_values:
        top_idx = np.argsort(repeat_score)[::-1][:k]
        aggregator["top_counts"][k][top_idx] += 1
        if k == topk_values[0]:
            for rank, chem_idx in enumerate(top_idx, start=1):
                aggregator["top_rows"].append(
                    {
                        "ranking_filter": filter_label,
                        "repeat": repeat,
                        "repeat_seed": repeat_seed,
                        "top_k": k,
                        "rank_in_repeat": rank,
                        "chemical_index": int(chem_idx),
                        "SMILES": data.chemicals[chem_idx],
                        "CAS": data.cas[chem_idx],
                        "repeat_oof_cancer_direction_delta_when_detected": float(repeat_score[chem_idx]),
                        "repeat_oof_cancer_direction_delta": float(
                            repeat_stats["OOF_CancerDirection_MeanDelta"][chem_idx]
                        ),
                    }
                )


def finalize_ranking_result(
    aggregator: Dict[str, object],
    *,
    filter_spec: Dict[str, object],
    topk_values: List[int],
) -> Dict[str, object]:
    eligible_repeat_count = int(aggregator["eligible_repeat_count"])
    ranking_stats = {}
    for key, value in aggregator.items():
        if key in {"top_counts", "eligible_repeat_count", "top_rows"}:
            continue
        ranking_stats[key] = value / max(eligible_repeat_count, 1)
    for k in topk_values:
        ranking_stats[f"Top{k}_Frequency"] = aggregator["top_counts"][k] / max(
            eligible_repeat_count, 1
        )
    return {
        "filter_label": str(filter_spec["label"]),
        "min_auc": filter_spec["min_auc"],
        "min_balanced_accuracy": filter_spec["min_balanced_accuracy"],
        "eligible_repeat_count": eligible_repeat_count,
        "ranking_stats": ranking_stats,
        "top_rows": aggregator["top_rows"],
    }


def summarize_ranking_filters(
    repeat_metrics_df: pd.DataFrame,
    filter_specs: List[Dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for spec in filter_specs:
        if repeat_metrics_df.empty:
            sub = repeat_metrics_df
        else:
            mask = repeat_metrics_df.apply(
                lambda row: repeat_passes_ranking_filter(row.to_dict(), spec), axis=1
            )
            sub = repeat_metrics_df.loc[mask]
        rows.append(
            {
                "ranking_filter": spec["label"],
                "min_auc": spec["min_auc"],
                "min_balanced_accuracy": spec["min_balanced_accuracy"],
                "n_repeats": int(sub.shape[0]),
                "auc_mean": float(sub["auc"].mean()) if not sub.empty else np.nan,
                "auc_min": float(sub["auc"].min()) if not sub.empty else np.nan,
                "auc_median": float(sub["auc"].median()) if not sub.empty else np.nan,
                "balanced_accuracy_mean": float(sub["balanced_accuracy"].mean()) if not sub.empty else np.nan,
                "balanced_accuracy_min": float(sub["balanced_accuracy"].min()) if not sub.empty else np.nan,
                "f1_mean": float(sub["f1"].mean()) if not sub.empty else np.nan,
                "accuracy_mean": float(sub["accuracy"].mean()) if not sub.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def repeat_ranking_diagnostics(meta: pd.DataFrame, repeat_score: np.ndarray) -> Dict[str, int]:
    order = np.argsort(repeat_score)[::-1]
    top10 = order[: min(10, len(order))]
    top20 = order[: min(20, len(order))]
    top50 = order[: min(50, len(order))]
    return {
        "zero_target_count_top10": int((meta.loc[top10, "Hazard_Target_Count"] == 0).sum()),
        "zero_target_count_top20": int((meta.loc[top20, "Hazard_Target_Count"] == 0).sum()),
        "zero_target_count_top50": int((meta.loc[top50, "Hazard_Target_Count"] == 0).sum()),
        "target_supported_count_top10": int((meta.loc[top10, "Hazard_Target_Count"] > 0).sum()),
        "target_supported_count_top20": int((meta.loc[top20, "Hazard_Target_Count"] > 0).sum()),
        "target_supported_count_top50": int((meta.loc[top50, "Hazard_Target_Count"] > 0).sum()),
    }


def build_repeat_ranking_rows(
    *,
    data,
    meta: pd.DataFrame,
    repeat_stats: Dict[str, np.ndarray],
    repeat_row: Dict[str, object],
    ranking_filter_specs: List[Dict[str, object]],
    top_n: int,
) -> List[Dict[str, object]]:
    repeat_score = repeat_stats[PRIMARY_RANKING_SCORE]
    order = np.argsort(repeat_score)[::-1]
    top_idx = order[: min(int(top_n), len(order))]
    passed_filters = [
        str(spec["label"])
        for spec in ranking_filter_specs
        if repeat_passes_ranking_filter(repeat_row, spec)
    ]
    rows = []
    for rank, chem_idx in enumerate(top_idx, start=1):
        rows.append(
            {
                "repeat": repeat_row["repeat"],
                "repeat_seed": repeat_row["repeat_seed"],
                "auc": repeat_row["auc"],
                "balanced_accuracy": repeat_row["balanced_accuracy"],
                "accuracy": repeat_row["accuracy"],
                "f1": repeat_row["f1"],
                "passed_ranking_filters": ";".join(passed_filters),
                "rank_in_repeat": rank,
                "chemical_index": int(chem_idx),
                "SMILES": data.chemicals[chem_idx],
                "CAS": data.cas[chem_idx],
                "repeat_oof_cancer_direction_delta_when_detected": float(
                    repeat_stats[PRIMARY_RANKING_SCORE][chem_idx]
                ),
                "repeat_oof_cancer_direction_delta": float(
                    repeat_stats["OOF_CancerDirection_MeanDelta"][chem_idx]
                ),
                "repeat_oof_signed_delta": float(repeat_stats["OOF_Population_MeanDelta"][chem_idx]),
                "repeat_oof_control_direction_delta": float(
                    repeat_stats["OOF_ControlDirection_MeanDelta"][chem_idx]
                ),
                "repeat_oof_signed_delta_when_detected": float(
                    repeat_stats["OOF_Conditional_MeanDelta_WhenDetected"][chem_idx]
                ),
                "repeat_oof_control_direction_delta_when_detected": float(
                    repeat_stats["OOF_ControlDirection_MeanDelta_WhenDetected"][chem_idx]
                ),
                "repeat_positive_delta_frequency_when_detected": float(
                    repeat_stats["OOF_Positive_Delta_Frequency_WhenDetected"][chem_idx]
                ),
                "repeat_cancer_group_positive_mean_delta_when_detected": float(
                    repeat_stats["OOF_CancerGroup_PositiveMeanDelta_WhenDetected"][chem_idx]
                ),
                "repeat_cancer_group_negative_mean_delta_when_detected": float(
                    repeat_stats["OOF_CancerGroup_NegativeMeanDelta_WhenDetected"][chem_idx]
                ),
                "repeat_cancer_group_positive_delta_frequency_when_detected": float(
                    repeat_stats["OOF_CancerGroup_PositiveDelta_Frequency_WhenDetected"][chem_idx]
                ),
                "repeat_control_group_positive_mean_delta_when_detected": float(
                    repeat_stats["OOF_ControlGroup_PositiveMeanDelta_WhenDetected"][chem_idx]
                ),
                "repeat_control_group_negative_mean_delta_when_detected": float(
                    repeat_stats["OOF_ControlGroup_NegativeMeanDelta_WhenDetected"][chem_idx]
                ),
                "repeat_control_group_positive_delta_frequency_when_detected": float(
                    repeat_stats["OOF_ControlGroup_PositiveDelta_Frequency_WhenDetected"][chem_idx]
                ),
                "Detection_Rate_File": float(meta.loc[chem_idx, "Detection_Rate_File"]),
                "Cancer_Detected_Count": int(meta.loc[chem_idx, "Cancer_Detected_Count"]),
                "Control_Detected_Count": int(meta.loc[chem_idx, "Control_Detected_Count"]),
                "Hazard_Target_Count": float(meta.loc[chem_idx, "Hazard_Target_Count"]),
            }
        )
    return rows


def run_repeated_mil_stability(data, args: argparse.Namespace) -> Dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for Exposure Set-MIL stability selection.") from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    y = data.y
    n_samples, n_chemicals = data.presence.shape
    topk_values = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]
    repeat_seeds = build_repeat_seeds(args)
    global_rate = data.detection_rate.astype(np.float32)
    x_all = make_numeric_tokens(data.presence, data.hazards, global_rate)
    meta = build_metadata(data)

    ranking_filter_specs, primary_filter_label = build_ranking_filter_specs(args)
    ranking_aggregators = {
        str(spec["label"]): init_ranking_aggregator(n_chemicals, topk_values)
        for spec in ranking_filter_specs
    }

    fold_metrics_rows = []
    repeat_metrics_rows = []
    prediction_rows = []
    repeat_ranking_rows = []
    counterfactual_validation_rows = []
    trained_model_count = 0

    for repeat, repeat_seed in enumerate(repeat_seeds, start=1):
        skf = StratifiedKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=repeat_seed,
        )
        repeat_prob = np.zeros(n_samples, dtype=np.float64)
        repeat_delta = np.zeros((n_samples, n_chemicals), dtype=np.float64)
        repeat_contribution = np.zeros((n_samples, n_chemicals), dtype=np.float64)
        repeat_fold_losses = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(data.presence, y), start=1):
            model, history, info = train_torch_model_fixed(
                x_all[train_idx],
                y[train_idx],
                seed=repeat_seed,
                id_embed_dim=args.id_embed_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=device,
            )
            history.to_csv(
                args.output_dir / f"mil_repeat_{repeat}_seed_{repeat_seed}_fold_{fold}_history.csv",
                index=False,
            )
            repeat_fold_losses.append(float(info["final_train_loss"]))
            trained_model_count += 1

            test_logits = torch_predict_logits(model, x_all[test_idx], args.batch_size, device)
            test_prob = 1.0 / (1.0 + np.exp(-test_logits))
            repeat_prob[test_idx] = test_prob.astype(np.float64)

            _, delta, _ = explain_counterfactual_exact(
                model,
                x_all[test_idx],
                batch_size=args.batch_size,
                device=device,
                classifier_chunk=args.classifier_chunk,
            )
            if repeat == 1 and fold == 1:
                validation_df = validate_counterfactual_exact(
                    model,
                    x_all[test_idx],
                    batch_size=args.batch_size,
                    device=device,
                    classifier_chunk=args.classifier_chunk,
                )
                if not validation_df.empty:
                    validation_df.insert(0, "repeat", repeat)
                    validation_df.insert(1, "repeat_seed", int(repeat_seed))
                    validation_df.insert(2, "fold", fold)
                    counterfactual_validation_rows.extend(validation_df.to_dict("records"))
            p_test = data.presence[test_idx].astype(np.float64)
            contribution = p_test * delta
            repeat_delta[test_idx] = delta.astype(np.float64)
            repeat_contribution[test_idx] = contribution.astype(np.float64)

            fold_row = binary_metrics(y[test_idx], test_prob)
            fold_row.update(
                {
                    "repeat": repeat,
                    "repeat_seed": int(repeat_seed),
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "epochs": int(args.epochs),
                    "final_train_loss": float(info["final_train_loss"]),
                }
            )
            fold_metrics_rows.append(fold_row)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "sample_id": data.sample_ids[test_idx],
                        "sample_index": test_idx,
                        "y_true": y[test_idx],
                        "prob_cancer": test_prob,
                        "repeat": repeat,
                        "repeat_seed": int(repeat_seed),
                        "fold": fold,
                    }
                )
            )

        repeat_row = binary_metrics(y, repeat_prob)
        repeat_stats = compute_repeat_stats(data, repeat_delta, repeat_contribution)
        repeat_score = repeat_stats[PRIMARY_RANKING_SCORE]
        repeat_row.update(
            {
                "repeat": repeat,
                "repeat_seed": int(repeat_seed),
                "n_splits": int(args.n_splits),
                "n_oof_samples": int(n_samples),
                "epochs": int(args.epochs),
                "final_train_loss_mean": float(np.mean(repeat_fold_losses)),
                "final_train_loss_median": float(np.median(repeat_fold_losses)),
                "ranking_primary_score": PRIMARY_RANKING_SCORE,
            }
        )
        repeat_row.update(repeat_ranking_diagnostics(meta, repeat_score))
        repeat_metrics_rows.append(repeat_row)
        repeat_ranking_rows.extend(
            build_repeat_ranking_rows(
                data=data,
                meta=meta,
                repeat_stats=repeat_stats,
                repeat_row=repeat_row,
                ranking_filter_specs=ranking_filter_specs,
                top_n=args.save_repeat_rankings_top_n,
            )
        )

        for spec in ranking_filter_specs:
            if repeat_passes_ranking_filter(repeat_row, spec):
                label = str(spec["label"])
                update_ranking_aggregator(
                    ranking_aggregators[label],
                    data=data,
                    repeat_stats=repeat_stats,
                    topk_values=topk_values,
                    filter_label=label,
                    repeat=repeat,
                    repeat_seed=int(repeat_seed),
                )

    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    repeat_metrics_df = pd.DataFrame(repeat_metrics_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    repeat_rankings_df = pd.DataFrame(repeat_ranking_rows)
    ranking_by_filter = {
        str(spec["label"]): finalize_ranking_result(
            ranking_aggregators[str(spec["label"])],
            filter_spec=spec,
            topk_values=topk_values,
        )
        for spec in ranking_filter_specs
    }
    top_membership_rows = []
    for result in ranking_by_filter.values():
        top_membership_rows.extend(result["top_rows"])

    return {
        "fold_metrics": fold_metrics_df,
        "repeat_metrics": repeat_metrics_df,
        "predictions": predictions_df,
        "repeat_rankings": repeat_rankings_df,
        "counterfactual_validation": pd.DataFrame(counterfactual_validation_rows),
        "top_membership": pd.DataFrame(top_membership_rows),
        "ranking_by_filter": ranking_by_filter,
        "ranking_filter_summary": summarize_ranking_filters(repeat_metrics_df, ranking_filter_specs),
        "primary_filter_label": primary_filter_label,
        "ranking_filter_specs": ranking_filter_specs,
        "trained_model_count": trained_model_count,
        "repeat_count": int(repeat_metrics_df.shape[0]),
        "topk_values": topk_values,
        "repeat_seeds": repeat_seeds,
    }


def build_stable_ranking(data, ranking_result: Dict[str, object]) -> pd.DataFrame:
    if int(ranking_result["eligible_repeat_count"]) < 1:
        raise ValueError(f"No repeats passed ranking filter: {ranking_result['filter_label']}")
    meta = build_metadata(data)
    stats = ranking_result["ranking_stats"]
    for key, value in stats.items():
        meta[key] = value
    meta["Ranking_Filter"] = ranking_result["filter_label"]
    meta["Ranking_Min_AUC"] = ranking_result["min_auc"]
    meta["Ranking_Min_Balanced_Accuracy"] = ranking_result["min_balanced_accuracy"]
    meta["Eligible_Repeat_Count"] = ranking_result["eligible_repeat_count"]

    meta[PRIMARY_PERCENTILE_COLUMN] = meta[PRIMARY_RANKING_SCORE].rank(pct=True)
    meta["OOF_CancerDirection_MeanDelta_PercentileRank"] = meta[
        "OOF_CancerDirection_MeanDelta"
    ].rank(pct=True)
    meta["OOF_Population_MeanDelta_PercentileRank"] = meta[
        "OOF_Population_MeanDelta"
    ].rank(pct=True)

    out = meta.sort_values(
        PRIMARY_RANKING_SCORE,
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(0, PRIMARY_RANK_COLUMN, np.arange(1, len(out) + 1))
    return out


def build_target_supported_ranking(
    ranking: pd.DataFrame,
    *,
    min_hazard_count: float,
) -> pd.DataFrame:
    out = ranking.loc[ranking["Hazard_Target_Count"] >= float(min_hazard_count)].copy()
    out = out.sort_values(
        PRIMARY_RANKING_SCORE,
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(0, TARGET_SUPPORTED_RANK_COLUMN, np.arange(1, len(out) + 1))
    return out


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("exposome_set_mil_stability_outputs_v9_level12"))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=10)
    parser.add_argument("--seed-step", type=int, default=10)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--top-k", default="20,50,100")
    parser.add_argument("--save-repeat-rankings-top-n", type=int, default=200)

    parser.add_argument("--id-embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--classifier-chunk", type=int, default=65536)
    parser.add_argument(
        "--ranking-min-auc",
        type=float,
        default=0.50,
        help="Minimum repeat-level OOF AUC required for primary chemical ranking. Use -1 to disable.",
    )
    parser.add_argument(
        "--ranking-min-balanced-accuracy",
        type=float,
        default=0.50,
        help="Minimum repeat-level OOF balanced accuracy required for primary chemical ranking. Use -1 to disable.",
    )
    parser.add_argument("--target-supported-min-hazard-count", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed_start)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_exposome_data_v9_level12(args.data_dir)
    pd.DataFrame([data.summary]).to_csv(args.output_dir / "prepared_data_summary.csv", index=False)
    build_metadata(data).to_csv(args.output_dir / "chemical_metadata_used.csv", index=False)
    ranking_filter_specs, primary_filter_label = build_ranking_filter_specs(args)
    repeat_seeds = build_repeat_seeds(args)

    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as fh:
        cfg = vars(args).copy()
        cfg["data_dir"] = str(cfg["data_dir"])
        cfg["output_dir"] = str(cfg["output_dir"])
        cfg["repeat_seeds"] = repeat_seeds
        cfg["chemical_ranking_primary_sort_column"] = PRIMARY_RANKING_SCORE
        cfg["chemical_ranking_uses_composite_score"] = False
        cfg["chemical_ranking_no_extra_weighting"] = True
        cfg["chemical_ranking_no_detection_rate_reweighting_after_attribution"] = True
        cfg["chemical_ranking_primary_score_definition"] = (
            "mean(max(delta_logit, 0)) among serum samples where the chemical was detected; "
            "negative delta values contribute 0 but remain in the detected-sample denominator"
        )
        cfg["chemical_ranking_attribution_level"] = "repeat_level_5fold_oof"
        cfg["chemical_ranking_primary_filter"] = primary_filter_label
        cfg["chemical_ranking_sensitivity_filters"] = ranking_filter_specs
        cfg["inner_validation_split"] = False
        cfg["fixed_epochs"] = int(args.epochs)
        cfg["detection_rate_channel"] = "global"
        cfg["global_rate_source"] = "Detection_Rate_File"
        cfg["positive_file"] = NTS_LEVEL12_POSITIVE_FILE
        cfg["negative_file"] = NTS_LEVEL12_NEGATIVE_FILE
        cfg["frequency_file"] = NTS_LEVEL12_FREQUENCY_FILE
        cfg["hazard_file"] = NTS_LEVEL12_HAZARD_FILE
        cfg["input_version"] = "v9_level12_20261009"
        json.dump(cfg, fh, ensure_ascii=False, indent=2)

    print("Prepared data summary:")
    print(json.dumps(data.summary, ensure_ascii=False, indent=2))
    print(f"Repeat seeds: {repeat_seeds}")

    stability = run_repeated_mil_stability(data, args)
    fold_metrics_df = stability["fold_metrics"]
    repeat_metrics_df = stability["repeat_metrics"]
    fold_metrics_summary = summarize_metrics(fold_metrics_df)
    repeat_metrics_summary = summarize_metrics(repeat_metrics_df)
    primary_result = stability["ranking_by_filter"][stability["primary_filter_label"]]
    ranking = build_stable_ranking(data, primary_result)
    target_supported_ranking = build_target_supported_ranking(
        ranking,
        min_hazard_count=args.target_supported_min_hazard_count,
    )

    fold_metrics_df.to_csv(args.output_dir / "mil_oof_performance_by_fold_model.csv", index=False)
    repeat_metrics_df.to_csv(args.output_dir / "mil_oof_performance_by_repeat_5fold.csv", index=False)
    fold_metrics_summary.to_csv(args.output_dir / "mil_oof_performance_summary_by_fold_model.csv", index=False)
    repeat_metrics_summary.to_csv(args.output_dir / "mil_oof_performance_summary_by_repeat_5fold.csv", index=False)
    stability["ranking_filter_summary"].to_csv(
        args.output_dir / "mil_ranking_quality_filter_summary_by_repeat.csv",
        index=False,
    )
    stability["predictions"].to_csv(args.output_dir / "mil_oof_predictions_by_repeat.csv", index=False)
    stability["repeat_rankings"].to_csv(
        args.output_dir / "mil_repeat_5fold_topN_chemical_rankings.csv",
        index=False,
    )
    if not stability["counterfactual_validation"].empty:
        stability["counterfactual_validation"].to_csv(
            args.output_dir / "counterfactual_exact_validation.csv",
            index=False,
        )
    repeat_diag_cols = [
        "repeat",
        "repeat_seed",
        "auc",
        "balanced_accuracy",
        "accuracy",
        "f1",
        "zero_target_count_top10",
        "zero_target_count_top20",
        "zero_target_count_top50",
        "target_supported_count_top10",
        "target_supported_count_top20",
        "target_supported_count_top50",
    ]
    repeat_metrics_df[repeat_diag_cols].to_csv(
        args.output_dir / "mil_repeat_5fold_ranking_diagnostics.csv",
        index=False,
    )
    stability["top_membership"].to_csv(
        args.output_dir / "mil_top_membership_by_repeat_5fold.csv",
        index=False,
    )

    ranking.to_csv(args.output_dir / "mil_stable_chemical_ranking.csv", index=False)
    ranking.head(200).to_csv(args.output_dir / "mil_stable_top200_chemicals.csv", index=False)
    target_supported_ranking.to_csv(
        args.output_dir / "mil_stable_target_supported_chemical_ranking.csv",
        index=False,
    )
    target_supported_ranking.head(200).to_csv(
        args.output_dir / "mil_stable_target_supported_top200_chemicals.csv",
        index=False,
    )

    for label, result in stability["ranking_by_filter"].items():
        if int(result["eligible_repeat_count"]) < 1:
            continue
        sensitivity_ranking = build_stable_ranking(data, result)
        sensitivity_target_supported = build_target_supported_ranking(
            sensitivity_ranking,
            min_hazard_count=args.target_supported_min_hazard_count,
        )
        sensitivity_ranking.to_csv(
            args.output_dir / f"mil_stable_chemical_ranking_{label}.csv",
            index=False,
        )
        sensitivity_ranking.head(200).to_csv(
            args.output_dir / f"mil_stable_top200_chemicals_{label}.csv",
            index=False,
        )
        sensitivity_target_supported.to_csv(
            args.output_dir / f"mil_stable_target_supported_chemical_ranking_{label}.csv",
            index=False,
        )
        sensitivity_target_supported.head(200).to_csv(
            args.output_dir / f"mil_stable_target_supported_top200_chemicals_{label}.csv",
            index=False,
        )

    print("\nRepeat-level 5-fold OOF performance summary:")
    print(repeat_metrics_summary)
    print("\nFold-model OOF performance summary:")
    print(fold_metrics_summary)
    print("\nRanking quality filter summary:")
    print(stability["ranking_filter_summary"])
    print(f"\nTrained fold models: {stability['trained_model_count']}")
    print(f"Evaluated repeat-level 5-fold rankings: {stability['repeat_count']}")
    print(f"Primary ranking filter: {stability['primary_filter_label']}")
    print(f"Eligible repeats for primary ranking: {primary_result['eligible_repeat_count']}")
    print(f"Primary ranking score: {PRIMARY_RANKING_SCORE}")
    print("Primary ranking uses no additional detection-rate, target-count, or composite weighting.")
    print("No inner validation split was used; all fold models use fixed epochs.")

    key_cols = [
        PRIMARY_RANK_COLUMN,
        "SMILES",
        "CAS",
        "Ranking_Filter",
        PRIMARY_RANKING_SCORE,
        "OOF_CancerDirection_MeanDelta",
        "OOF_Population_MeanDelta",
        "OOF_ControlDirection_MeanDelta",
        "OOF_CancerGroup_PositiveMeanDelta_WhenDetected",
        "OOF_CancerGroup_NegativeMeanDelta_WhenDetected",
        "OOF_CancerGroup_PositiveDelta_Frequency_WhenDetected",
        "OOF_ControlGroup_PositiveMeanDelta_WhenDetected",
        "OOF_ControlGroup_NegativeMeanDelta_WhenDetected",
        "OOF_ControlGroup_PositiveDelta_Frequency_WhenDetected",
        "OOF_Positive_Delta_Frequency_WhenDetected",
        "Top20_Frequency" if "Top20_Frequency" in ranking.columns else ranking.columns[-1],
        "Top50_Frequency" if "Top50_Frequency" in ranking.columns else ranking.columns[-1],
        "Cancer_Detected_Count",
        "Control_Detected_Count",
        "Hazard_Target_Count",
    ]
    print("\nTop chemicals by cancer-direction OOF contribution:")
    print(ranking[key_cols].head(30))

    target_cols = [TARGET_SUPPORTED_RANK_COLUMN] + key_cols
    target_cols = [col for col in target_cols if col in target_supported_ranking.columns]
    print("\nTop target-supported chemicals by cancer-direction OOF contribution:")
    print(target_supported_ranking[target_cols].head(30))
    print(f"\nOutputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

