"""
Gated chemical-token MIL pipeline for cohort-specific cancer potential ranking.

Expected input files in --data-dir:
  - all_results_Exogenous_ranking_binary.csv
      SMILES + 10 binary hazard/target columns.
  - Exogenous_detection_frequency_exposure.csv
      SMILES, Detected_Samples, CAS, Detection_Rate.
  - diptox_Positive_sample_4th_all.csv
      Long table of detected chemicals in positive/cancer samples.
  - diptox_Negative_sample_4th_all.csv
      Long table of detected chemicals in negative/control samples.

The PyTorch model is optional. This environment may not have torch installed;
use --mode prepare or --mode elasticnet to verify data handling without torch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


HAZARD_FILE = "all_results_Exogenous_ranking_binary.csv"
FREQUENCY_FILE = "Exogenous_detection_frequency_exposure.csv"
POSITIVE_FILE = "diptox_Positive_sample_4th_all.csv"
NEGATIVE_FILE = "diptox_Negative_sample_4th_all.csv"


@dataclass
class ExposomeData:
    sample_ids: np.ndarray
    y: np.ndarray
    chemicals: np.ndarray
    cas: np.ndarray
    target_names: List[str]
    presence: np.ndarray
    hazards: np.ndarray
    detection_rate: np.ndarray
    summary: Dict[str, object]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def clean_smiles(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    return out


def load_exposome_data(data_dir: Path) -> ExposomeData:
    """
    Build sample x chemical presence, hazard and detection-rate matrices.
    """
    hazard_df = read_csv_checked(data_dir / HAZARD_FILE).copy()
    freq_df = read_csv_checked(data_dir / FREQUENCY_FILE).copy()
    pos_df = read_csv_checked(data_dir / POSITIVE_FILE).copy()
    neg_df = read_csv_checked(data_dir / NEGATIVE_FILE).copy()

    required_sample_cols = {"CAS", "ID", "cano_smiles", "active"}
    for name, df in [(POSITIVE_FILE, pos_df), (NEGATIVE_FILE, neg_df)]:
        missing = required_sample_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} misses columns: {sorted(missing)}")

    if "SMILES" not in hazard_df.columns:
        raise ValueError(f"{HAZARD_FILE} must contain SMILES.")
    if not {"SMILES", "Detection_Rate", "CAS"}.issubset(freq_df.columns):
        raise ValueError(f"{FREQUENCY_FILE} must contain SMILES, CAS, Detection_Rate.")

    hazard_df["SMILES"] = clean_smiles(hazard_df["SMILES"])
    freq_df["SMILES"] = clean_smiles(freq_df["SMILES"])
    pos_df["cano_smiles"] = clean_smiles(pos_df["cano_smiles"])
    neg_df["cano_smiles"] = clean_smiles(neg_df["cano_smiles"])

    sample_df = pd.concat([pos_df, neg_df], ignore_index=True)
    sample_df = sample_df.dropna(subset=["ID", "cano_smiles", "active"])
    sample_df["active"] = sample_df["active"].astype(int)

    label_nunique = sample_df.groupby("ID")["active"].nunique()
    conflict_ids = label_nunique[label_nunique > 1].index.tolist()
    if conflict_ids:
        raise ValueError(f"Sample IDs with conflicting labels: {conflict_ids[:10]}")

    label_by_id = sample_df.groupby("ID", sort=True)["active"].first()
    sample_ids = label_by_id.index.to_numpy()
    y = label_by_id.to_numpy(dtype=np.int64)

    target_names = [c for c in hazard_df.columns if c != "SMILES"]
    if len(target_names) != 10:
        print(
            f"Warning: expected 10 hazard target columns, found {len(target_names)}: {target_names}",
            file=sys.stderr,
        )

    sample_chems = set(sample_df["cano_smiles"].dropna())
    hazard_chems = set(hazard_df["SMILES"].dropna())
    freq_chems = set(freq_df["SMILES"].dropna())

    # The sample tables include 100 endogenous chemicals that were deliberately
    # removed from the exogenous detection-rate and hazard-ranking files. The
    # model ranks only exogenous chemicals present in all three sources.
    selected = sample_chems & hazard_chems & freq_chems
    chemicals = [s for s in freq_df["SMILES"].tolist() if s in selected]

    if not chemicals:
        raise ValueError("No usable chemicals after alignment.")

    chem_to_col = {s: j for j, s in enumerate(chemicals)}
    id_to_row = {sid: i for i, sid in enumerate(sample_ids)}

    presence = np.zeros((len(sample_ids), len(chemicals)), dtype=np.float32)
    selected_rows = sample_df[sample_df["cano_smiles"].isin(chem_to_col)]
    duplicate_id_smiles = int(selected_rows.duplicated(["ID", "cano_smiles"]).sum())

    for sid, smi in selected_rows[["ID", "cano_smiles"]].itertuples(index=False, name=None):
        presence[id_to_row[sid], chem_to_col[smi]] = 1.0

    hazard_aligned = (
        hazard_df.drop_duplicates("SMILES")
        .set_index("SMILES")
        .reindex(chemicals)[target_names]
        .fillna(0.0)
    )
    hazards = hazard_aligned.to_numpy(dtype=np.float32)

    freq_aligned = (
        freq_df.drop_duplicates("SMILES")
        .set_index("SMILES")
        .reindex(chemicals)
    )
    detection_rate = freq_aligned["Detection_Rate"].to_numpy(dtype=np.float32)

    cas = freq_aligned["CAS"].fillna("").astype(str).to_numpy()

    summary = {
        "n_samples": int(len(sample_ids)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_sample_unique_chemicals": int(len(sample_chems)),
        "n_hazard_unique_chemicals": int(len(hazard_chems)),
        "n_frequency_unique_chemicals": int(len(freq_chems)),
        "n_selected_chemicals": int(len(chemicals)),
        "sample_hazard_frequency_intersection": int(len(sample_chems & hazard_chems & freq_chems)),
        "excluded_sample_chemicals_not_in_exogenous_feature_tables": int(len(sample_chems - (hazard_chems & freq_chems))),
        "duplicate_id_smiles_rows_collapsed": duplicate_id_smiles,
        "mean_detected_per_sample_after_alignment": float(presence.sum(axis=1).mean()),
        "min_detected_per_sample_after_alignment": int(presence.sum(axis=1).min()),
        "max_detected_per_sample_after_alignment": int(presence.sum(axis=1).max()),
    }

    return ExposomeData(
        sample_ids=sample_ids,
        y=y,
        chemicals=np.array(chemicals, dtype=object),
        cas=cas,
        target_names=target_names,
        presence=presence,
        hazards=hazards,
        detection_rate=detection_rate,
        summary=summary,
    )


def make_numeric_tokens(
    presence: np.ndarray,
    hazards: np.ndarray,
    detection_rate: np.ndarray,
) -> np.ndarray:
    """
    Create gated numeric token features:
      [presence, presence*detection_rate, presence*target_1, ..., presence*target_10]
    Shape: n_samples x n_chemicals x 12.
    """
    p = presence.astype(np.float32, copy=False)
    n, c = p.shape
    t = hazards.shape[1]
    x = np.zeros((n, c, 2 + t), dtype=np.float32)
    x[:, :, 0] = p
    x[:, :, 1] = p * detection_rate.reshape(1, c)
    x[:, :, 2:] = p[:, :, None] * hazards.reshape(1, c, t)
    return x


def save_prepared_outputs(data: ExposomeData, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([data.summary]).to_csv(output_dir / "prepared_data_summary.csv", index=False)

    meta = pd.DataFrame(
        {
            "chemical_index": np.arange(len(data.chemicals)),
            "SMILES": data.chemicals,
            "CAS": data.cas,
            "Detection_Rate": data.detection_rate,
        }
    )
    for k, name in enumerate(data.target_names):
        meta[name] = data.hazards[:, k]
    meta["Detected_Count_In_100"] = data.presence.sum(axis=0).astype(int)
    meta.to_csv(output_dir / "chemical_metadata_used.csv", index=False)

    sample_meta = pd.DataFrame({"sample_id": data.sample_ids, "label": data.y})
    sample_meta["n_detected_used"] = data.presence.sum(axis=1).astype(int)
    sample_meta.to_csv(output_dir / "sample_metadata_used.csv", index=False)

    np.savez_compressed(
        output_dir / "prepared_exposome_data.npz",
        sample_ids=data.sample_ids.astype(str),
        y=data.y,
        chemicals=data.chemicals.astype(str),
        cas=data.cas.astype(str),
        presence=data.presence.astype(np.uint8),
        hazards=data.hazards.astype(np.float32),
        detection_rate=data.detection_rate.astype(np.float32),
        target_names=np.array(data.target_names, dtype=object),
    )


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, prob))
    else:
        out["auc"] = float("nan")
    return out


def run_elasticnet_baseline(data: ExposomeData, output_dir: Path, seed: int, n_splits: int) -> None:
    """
    Sparse baseline on the raw presence matrix. This is not the proposed final
    model, but it is an important low-variance benchmark for small n.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    x = data.presence
    y = data.y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    coef_sum = np.zeros(x.shape[1], dtype=np.float64)

    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.2,
            class_weight="balanced",
            max_iter=10000,
            random_state=seed + fold,
            n_jobs=1,
        )
        clf.fit(x[train_idx], y[train_idx])
        prob = clf.predict_proba(x[test_idx])[:, 1]
        m = binary_metrics(y[test_idx], prob)
        m.update({"fold": fold, "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
        rows.append(m)
        coef_sum += clf.coef_.ravel()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "elasticnet_cv_metrics.csv", index=False)

    coef_mean = coef_sum / n_splits
    coef_table = pd.DataFrame(
        {
            "SMILES": data.chemicals,
            "CAS": data.cas,
            "elasticnet_mean_coef": coef_mean,
            "elasticnet_abs_coef": np.abs(coef_mean),
            "Detected_Count_In_100": data.presence.sum(axis=0).astype(int),
        }
    )
    for k, name in enumerate(data.target_names):
        coef_table[name] = data.hazards[:, k]
    coef_table = coef_table.sort_values("elasticnet_abs_coef", ascending=False)
    coef_table.to_csv(output_dir / "elasticnet_chemical_coefficients.csv", index=False)


def import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed in this Python environment. "
            "Run --mode prepare or --mode elasticnet now, or install torch before --mode torch."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def build_torch_classes():
    torch, nn, DataLoader, TensorDataset = import_torch()

    class GatedAttentionMIL(nn.Module):
        """
        Efficient chemical-token MIL model.

        It avoids full 5674 x 5674 self-attention. Each detected chemical token is
        encoded, attention pooling summarizes the detected set, and the classifier
        predicts cancer probability. Masked absent chemicals do not participate in
        attention pooling.
        """

        def __init__(
            self,
            num_chemicals: int,
            numeric_dim: int,
            id_embed_dim: int = 8,
            hidden_dim: int = 32,
            dropout: float = 0.4,
        ):
            super().__init__()
            self.num_chemicals = num_chemicals
            self.numeric_dim = numeric_dim
            self.id_embed_dim = id_embed_dim
            self.hidden_dim = hidden_dim

            self.id_embedding = nn.Embedding(num_chemicals, id_embed_dim)
            self.token_encoder = nn.Sequential(
                nn.Linear(numeric_dim + id_embed_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.attn_v = nn.Linear(hidden_dim, hidden_dim)
            self.attn_u = nn.Linear(hidden_dim, hidden_dim)
            self.attn_w = nn.Linear(hidden_dim, 1, bias=False)
            mid = max(8, hidden_dim // 2)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, mid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid, 1),
            )

        def forward(self, x, return_parts: bool = False):
            # x: batch x chemicals x numeric_dim
            b, c, _ = x.shape
            presence = x[:, :, 0]
            mask = presence > 0

            ids = torch.arange(c, device=x.device)
            emb = self.id_embedding(ids).unsqueeze(0).expand(b, c, self.id_embed_dim)
            emb = emb * presence.unsqueeze(-1)

            token_in = torch.cat([x, emb], dim=-1)
            h = self.token_encoder(token_in)

            score = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(-1)
            has_token = mask.any(dim=1, keepdim=True)
            score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
            attn = torch.softmax(score, dim=1)
            attn = torch.where(has_token, attn, torch.zeros_like(attn))
            z = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
            logits = self.classifier(z).squeeze(-1)

            if return_parts:
                return {"logits": logits, "h": h, "scores": score, "attn": attn, "z": z, "mask": mask}
            return logits

    return torch, nn, DataLoader, TensorDataset, GatedAttentionMIL


def torch_predict_logits(model, x_np: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    torch, _, DataLoader, TensorDataset = import_torch()
    model.eval()
    ds = TensorDataset(torch.tensor(x_np, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logits = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits.append(model(xb).detach().cpu().numpy())
    return np.concatenate(logits, axis=0)


def train_torch_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    id_embed_dim: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
):
    torch, nn, DataLoader, TensorDataset, GatedAttentionMIL = build_torch_classes()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    best_state = None
    best_score = -np.inf
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
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

        val_logits = torch_predict_logits(model, x_val, batch_size=batch_size, device=device)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        if len(np.unique(y_val)) == 2:
            val_score = roc_auc_score(y_val, val_prob)
        else:
            val_score = -float(np.mean((val_prob - y_val) ** 2))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_score": float(val_score),
            }
        )

        if val_score > best_score + 1e-5:
            best_score = float(val_score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val_score": best_score}


def run_torch_cv(data: ExposomeData, output_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, _, _, _ = import_torch()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    y = data.y
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows = []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(data.presence, y), start=1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed + fold)
        rel_train_idx, rel_val_idx = next(sss.split(data.presence[train_val_idx], y[train_val_idx]))
        train_idx = train_val_idx[rel_train_idx]
        val_idx = train_val_idx[rel_val_idx]

        # Strict evaluation: detection rate is estimated from training samples only.
        fold_rate = data.presence[train_idx].mean(axis=0).astype(np.float32)
        x_all = make_numeric_tokens(data.presence, data.hazards, fold_rate)

        model, history, info = train_torch_model(
            x_all[train_idx],
            y[train_idx],
            x_all[val_idx],
            y[val_idx],
            seed=args.seed + fold,
            id_embed_dim=args.id_embed_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=device,
        )
        history.to_csv(output_dir / f"torch_fold_{fold}_history.csv", index=False)

        test_logits = torch_predict_logits(model, x_all[test_idx], args.batch_size, device)
        test_prob = 1.0 / (1.0 + np.exp(-test_logits))
        m = binary_metrics(y[test_idx], test_prob)
        m.update(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "best_epoch": int(info["best_epoch"]),
                "best_val_score": float(info["best_val_score"]),
            }
        )
        rows.append(m)

        pred_df = pd.DataFrame(
            {
                "sample_id": data.sample_ids[test_idx],
                "y_true": y[test_idx],
                "prob_cancer": test_prob,
                "fold": fold,
            }
        )
        pred_df.to_csv(output_dir / f"torch_fold_{fold}_test_predictions.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "torch_cv_metrics.csv", index=False)
    return metrics


def explain_counterfactual_exact(
    model,
    x_np: np.ndarray,
    batch_size: int,
    device: str,
    classifier_chunk: int = 65536,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact chemical deletion for GatedAttentionMIL.

    Returns:
      base_logits: n_samples
      delta_logits: n_samples x n_chemicals
      attention: n_samples x n_chemicals
    """
    torch, _, DataLoader, TensorDataset = import_torch()
    model.eval()
    ds = TensorDataset(torch.tensor(x_np, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    base_all = []
    delta_all = []
    attn_all = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            parts = model(xb, return_parts=True)
            h = parts["h"]
            score = parts["scores"]
            mask = parts["mask"].float()

            row_max = torch.where(mask > 0, score, torch.tensor(-1e30, device=device)).max(dim=1, keepdim=True).values
            exp_score = torch.exp(score - row_max) * mask
            den = exp_score.sum(dim=1, keepdim=True).clamp_min(1e-12)
            num = torch.einsum("bc,bch->bh", exp_score, h)
            z = num / den
            base_logits = model.classifier(z).squeeze(-1)

            # Delete each detected chemical from the attention pool.
            den_wo = (den - exp_score).clamp_min(1e-12)
            num_wo = num[:, None, :] - exp_score[:, :, None] * h
            z_wo = num_wo / den_wo[:, :, None]

            flat = z_wo.reshape(-1, z_wo.shape[-1])
            logits_wo_chunks = []
            for start in range(0, flat.shape[0], classifier_chunk):
                logits_wo_chunks.append(model.classifier(flat[start : start + classifier_chunk]).squeeze(-1))
            logits_wo = torch.cat(logits_wo_chunks, dim=0).reshape(xb.shape[0], xb.shape[1])

            delta = (base_logits[:, None] - logits_wo) * mask
            attn = (exp_score / den) * mask

            base_all.append(base_logits.detach().cpu().numpy())
            delta_all.append(delta.detach().cpu().numpy())
            attn_all.append(attn.detach().cpu().numpy())

    return (
        np.concatenate(base_all, axis=0),
        np.concatenate(delta_all, axis=0),
        np.concatenate(attn_all, axis=0),
    )


def train_final_and_explain(data: ExposomeData, output_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, _, _, _ = import_torch()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Final cohort model: use the supplied cohort detection-rate file.
    x_all = make_numeric_tokens(data.presence, data.hazards, data.detection_rate)
    y = data.y
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed + 991)
    train_idx, val_idx = next(sss.split(x_all, y))

    model, history, info = train_torch_model(
        x_all[train_idx],
        y[train_idx],
        x_all[val_idx],
        y[val_idx],
        seed=args.seed + 991,
        id_embed_dim=args.id_embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
    )
    history.to_csv(output_dir / "final_model_history.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "num_chemicals": int(x_all.shape[1]),
                "numeric_dim": int(x_all.shape[2]),
                "id_embed_dim": args.id_embed_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "chemicals": data.chemicals.astype(str).tolist(),
            "target_names": data.target_names,
            "training_info": info,
        },
        output_dir / "final_gated_attention_mil_model.pt",
    )

    logits = torch_predict_logits(model, x_all, args.batch_size, device)
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred_df = pd.DataFrame({"sample_id": data.sample_ids, "label": y, "prob_cancer": prob, "logit": logits})
    pred_df.to_csv(output_dir / "final_model_sample_predictions.csv", index=False)

    base_logits, delta, attn = explain_counterfactual_exact(
        model,
        x_all,
        batch_size=args.batch_size,
        device=device,
        classifier_chunk=args.classifier_chunk,
    )
    np.savez_compressed(
        output_dir / "counterfactual_delta_logits.npz",
        sample_ids=data.sample_ids.astype(str),
        chemicals=data.chemicals.astype(str),
        base_logits=base_logits,
        delta_logits=delta,
        attention=attn,
    )

    p = data.presence
    detected_count = p.sum(axis=0)
    cancer_mask = y == 1
    control_mask = y == 0
    cancer_detected = p[cancer_mask].sum(axis=0)
    control_detected = p[control_mask].sum(axis=0)
    contribution = p * delta
    denom = np.maximum(detected_count, 1.0)

    scores = pd.DataFrame(
        {
            "chemical_index": np.arange(len(data.chemicals)),
            "SMILES": data.chemicals,
            "CAS": data.cas,
            "Detection_Rate_File": data.detection_rate,
            "Detected_Count_In_100": detected_count.astype(int),
            "Cancer_Detected_Count": cancer_detected.astype(int),
            "Control_Detected_Count": control_detected.astype(int),
            "Cancer_Detection_Rate": cancer_detected / max(cancer_mask.sum(), 1),
            "Control_Detection_Rate": control_detected / max(control_mask.sum(), 1),
            "Population_Potential_MeanDelta": contribution.mean(axis=0),
            "Conditional_Potency_MeanDelta": contribution.sum(axis=0) / denom,
            "Mean_Attention_When_Detected": (attn * p).sum(axis=0) / denom,
            "Positive_Delta_Sample_Count": ((delta > 0) & (p > 0)).sum(axis=0).astype(int),
        }
    )
    for k, name in enumerate(data.target_names):
        scores[name] = data.hazards[:, k]
    scores["Hazard_Target_Count"] = data.hazards.sum(axis=1)
    scores = scores.sort_values(
        ["Population_Potential_MeanDelta", "Conditional_Potency_MeanDelta"],
        ascending=False,
    )
    scores.to_csv(output_dir / "chemical_carcinogenic_potential_scores.csv", index=False)
    return scores


def run_bootstrap_stability(data: ExposomeData, output_dir: Path, args: argparse.Namespace) -> None:
    if args.bootstrap_runs <= 0:
        return
    torch, _, _, _ = import_torch()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)
    x_all = make_numeric_tokens(data.presence, data.hazards, data.detection_rate)
    y = data.y
    rng = np.random.default_rng(args.seed + 2026)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    top_counts = np.zeros(len(data.chemicals), dtype=np.int64)

    for b in range(1, args.bootstrap_runs + 1):
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])
        rng.shuffle(boot_idx)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed + 3000 + b)
        rel_train, rel_val = next(sss.split(x_all[boot_idx], y[boot_idx]))
        train_idx = boot_idx[rel_train]
        val_idx = boot_idx[rel_val]

        model, _, _ = train_torch_model(
            x_all[train_idx],
            y[train_idx],
            x_all[val_idx],
            y[val_idx],
            seed=args.seed + 3000 + b,
            id_embed_dim=args.id_embed_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=device,
        )
        _, delta, _ = explain_counterfactual_exact(
            model,
            x_all,
            batch_size=args.batch_size,
            device=device,
            classifier_chunk=args.classifier_chunk,
        )
        score = (data.presence * delta).mean(axis=0)
        top = np.argsort(score)[::-1][: args.bootstrap_top_k]
        top_counts[top] += 1

        pd.DataFrame(
            {"bootstrap_run": [b], "completed": [True], "top_k": [args.bootstrap_top_k]}
        ).to_csv(output_dir / "bootstrap_progress.csv", mode="a", header=(b == 1), index=False)

    stability = pd.DataFrame(
        {
            "SMILES": data.chemicals,
            "CAS": data.cas,
            "top_k_frequency": top_counts / max(args.bootstrap_runs, 1),
            "top_k_count": top_counts,
        }
    ).sort_values("top_k_frequency", ascending=False)
    stability.to_csv(output_dir / "bootstrap_topk_stability.csv", index=False)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gated chemical-token MIL pipeline.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing the four input CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("exposome_set_mil_outputs"))
    parser.add_argument("--mode", choices=["prepare", "elasticnet", "torch"], default="prepare")
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--n-splits", type=int, default=5)

    # Torch model hyperparameters.
    parser.add_argument("--id-embed-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--classifier-chunk", type=int, default=65536)
    parser.add_argument("--skip-cv", action="store_true", help="In torch mode, skip outer CV and train/explain final model only.")
    parser.add_argument("--skip-final", action="store_true", help="In torch mode, skip final model and explanation.")
    parser.add_argument("--bootstrap-runs", type=int, default=0)
    parser.add_argument("--bootstrap-top-k", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_exposome_data(args.data_dir)
    save_prepared_outputs(data, args.output_dir)

    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as fh:
        cfg = vars(args).copy()
        cfg["data_dir"] = str(cfg["data_dir"])
        cfg["output_dir"] = str(cfg["output_dir"])
        json.dump(cfg, fh, ensure_ascii=False, indent=2)

    print("Prepared data summary:")
    print(json.dumps(data.summary, ensure_ascii=False, indent=2))

    if args.mode == "prepare":
        print(f"Prepared files written to: {args.output_dir.resolve()}")
        return

    if args.mode == "elasticnet":
        run_elasticnet_baseline(data, args.output_dir, seed=args.seed, n_splits=args.n_splits)
        print(f"Elastic Net baseline written to: {args.output_dir.resolve()}")
        return

    if args.mode == "torch":
        if not args.skip_cv:
            cv_metrics = run_torch_cv(data, args.output_dir, args)
            print("Torch CV metrics:")
            print(cv_metrics)
        if not args.skip_final:
            scores = train_final_and_explain(data, args.output_dir, args)
            print("Top chemical potential scores:")
            print(scores.head(20))
        run_bootstrap_stability(data, args.output_dir, args)
        print(f"Torch outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
