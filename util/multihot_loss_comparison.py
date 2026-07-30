"""
Compare ZeroInflatedMultiLabelLoss vs weighted BCE vs focal loss on SOC multi-hot training.

Designed to run from Agentic_LanceDB_alpha_5_24.ipynb after tensors, loaders, and
SOCMultiHotModel are defined. Example:

    from loss_experiments.multihot_loss_comparison import run_loss_comparison_experiment

    loss_cmp_df, loss_cmp_runs = run_loss_comparison_experiment(
        train_loader=train_loader_1hot,
        val_loader=val_loader_1hot,device = "cuda" if torch.cuda.is_available() else "cpu"
        model_factory=lambda: SOCMultiHotModel(
            token_dim=TOKEN_DIM,
            hidden=128,
            num_soc_classes=num_soc_classes,
            dropout=0.2,
            prior_logits=soc_prior_logits,
            dual_branch=True,
        ),
        prior_logits=soc_prior_logits,
        pos_weight=soc_pos_weight,
        device=device,
        eval_multihot_loader=eval_multihot_loader,
        collect_multihot_probs=collect_multihot_probs,
        tune_threshold_fn=tune_multihot_threshold_recall_floor,
        num_epochs=40,
        patience=5,
    )
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, asdict, is_dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Loss functions (BCE / Focal baselines + notebook ASL alias)
# ---------------------------------------------------------------------------
from sklearn.metrics import f1_score, recall_score, precision_score

MULTIHOT_THRESHOLD = 0.3  # initial threshold for prob -> 0/1 guesses; will be tuned based on val set

def eval_multihot_loader(model, loader, criterion, threshold=MULTIHOT_THRESHOLD):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []

    # Logit-adjusted losses (see ZeroInflatedMultiLabelLoss(pos_rate=..., tau=...))
    # deliberately shift each class's raw output scale -- an uninformative,
    # base-rate example now outputs sigmoid ~= 0.5 instead of ~= pos_rate. A flat
    # threshold tuned for an unadjusted model (e.g. the default 0.3) then fires on
    # nearly everything, which tanks precision here *and* corrupts early-stopping/
    # checkpoint selection since this same function drives `best_val_f1` in
    # train_one_loss. Translate `threshold` into the equivalent per-class cutoff
    # in the adjusted output space instead of comparing against the flat value.
    decode_threshold = threshold
    if hasattr(criterion, "logit_prior") and getattr(criterion, "tau", 0) != 0:
        logit_thr = torch.log(torch.as_tensor(threshold / (1.0 - threshold)))
        decode_threshold = torch.sigmoid(
            logit_thr.to(criterion.logit_prior.device)
            - criterion.tau * criterion.logit_prior
        )  # per-class tensor; broadcasts against (batch, C) predictions

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)  # ✓ Always scalar now
            pred = (torch.sigmoid(logits) >= decode_threshold).float()
            bs = x_batch.size(0)
            total_loss += loss.item() * bs
            n += bs
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    return (
        total_loss / n,
        f1_score(y_true, y_pred, average="macro", zero_division=0),
        recall_score(y_true, y_pred, average="macro", zero_division=0),
        y_pred.mean(),
    )

def collect_multihot_probs(model, loader):
    model.eval()
    y_true, probs = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(
                y_batch.numpy() if isinstance(y_batch, np.ndarray) else y_batch.cpu().numpy()
            )
    return np.vstack(y_true), np.vstack(probs)

def multihot_metrics(y_true, y_pred, label=""):

    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(
        f"{label} macro — P={prec:.4f} R={rec:.4f} F1={f1:.4f} | "
        f"pred density={y_pred.mean():.4f}"
    )
    return prec, rec, f1

def tune_multihot_threshold_recall_floor(
    y_true, probs, precision_floor=0.3, thresholds=None
):
    """Maximize macro recall subject to macro precision >= precision_floor."""
    thresholds = thresholds or np.arange(0.08, 0.55, 0.02)

    # initiate defaults
    best_threshold, best_rec, best_prec = MULTIHOT_THRESHOLD, -1.0, 0.0
    rows = []
    for t in thresholds:
        pred = (probs >= t).astype(float)
        prec = precision_score(y_true, pred, average="macro", zero_division=0)
        rec = recall_score(y_true, pred, average="macro", zero_division=0)
        rows.append((t, prec, rec))

        #update best if meets precision floor and improves recall
        if prec >= precision_floor and rec > best_rec:
            best_rec, best_prec, best_threshold = rec, prec, float(t)

    if best_rec < 0:
        # fallback: highest recall at any threshold
        for t, prec, rec in rows:
            if rec > best_rec:
                best_rec, best_prec, best_threshold = rec, prec, float(t)
        print(f"Warning: no threshold met precision floor {precision_floor}; using max-recall fallback.")
    return best_threshold, best_prec, best_rec, rows

# def tune_multihot_threshold_f1(model, loader, thresholds=None):
#     # thresholds = thresholds or np.arange(0.15, 0.55, 0.05)
#     model.eval()
#     y_true, probs = [], []
#     with torch.no_grad():
#         for x_batch, y_batch in loader:
#             logits = model(x_batch.to(device))
#             probs.append(torch.sigmoid(logits).cpu().numpy())
#             y_true.append(y_batch.numpy())

#     y_true = np.vstack(y_true)
#     probs = np.vstack(probs)
    
#     #placeholder for 0.35 or other initial guess
#     best_threshold, best_f1 = MULTIHOT_THRESHOLD, 0.0
#     for t in (thresholds or np.arange(0.15, 0.55, 0.05)):
#         f1 = f1_score(y_true, (probs >= t).astype(float), average="macro", zero_division=0)
#         if f1 > best_f1:
#             best_f1, best_threshold = f1, float(t)
#     return best_threshold, best_f1

def tune_multihot_threshold_f1(y_true, probs, thresholds=None):
    
    #placeholder for 0.35 or other initial guess
    best_threshold, best_f1 = 0.3, 0.0
    for t in (thresholds or np.arange(0.15, 0.55, 0.05)):
        f1 = f1_score(y_true, (probs >= t).astype(float), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)
            
    return best_threshold, best_f1
     
    
class SOCMultiHotModel(nn.Module):
    """
    Multi-hot SOC classifier over 3 retrieved PT embedding tokens.

    Input:  (B, 3, 1024) — one normalized PT embedding per token
    Output: (B, num_soc_classes) logits (default 27 SOCs)
    """
    NUM_TOKENS = 3
    TOKEN_DIM = 1024

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        hidden: int = 768,
        num_soc_classes: int = 27,
        dropout: float = 0.2,
        prior_logits=None,
        num_heads: int = 4,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden = int(hidden)
        self.num_soc_classes = int(num_soc_classes)
        self.num_heads = int(num_heads)

        self.token_encoder = nn.Sequential(
            nn.Linear(self.token_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden)
        self.pool_attn = nn.Linear(hidden, 1)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_soc_classes),
        )
        self._reset_parameters(prior_logits)

    def _reset_parameters(self, prior_logits=None):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if prior_logits is not None:
            with torch.no_grad():
                self.head[-1].bias.copy_(prior_logits.to(self.head[-1].bias.device))

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Accept (B, 3, 1024); also (B, 10, D) by keeping first 3 tokens and first 1024 dims."""
        if x.dim() == 2:
            x = x.view(x.size(0), self.NUM_TOKENS, -1)
        if x.size(1) > self.NUM_TOKENS:
            x = x[:, : self.NUM_TOKENS, :]
        if x.size(-1) > self.token_dim:
            x = x[..., : self.token_dim]
        elif x.size(-1) < self.token_dim:
            raise ValueError(
                f"Expected token_dim>={self.token_dim}, got last dim {x.size(-1)}"
            )
        return x

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x = self._prepare_tokens(x)
        h = self.token_encoder(x)  # (B, 3, hidden)

        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        h = self.attn_norm(h + attn_out)

        scores = self.pool_attn(h).squeeze(-1)
        w = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(w.unsqueeze(1), h).squeeze(1)

        logits = self.head(pooled)
        if return_attn:
            return logits, w
        return logits

class SOCLabelQueryModel(nn.Module):
    """
    Each SOC has a learnable query that cross-attends over the token embeddings;
    a group-wise head turns each label's attended vector into its own logit.
    Same __init__/forward contract as SOCMultiHotModel (usable via model_factory).
    Returns (logits, attn) with return_attn=True; attn is (B, num_labels, T) —
    per-label token attribution, useful for interpreting which token drove a SOC.
    Accepts an optional key_padding_mask so T can vary per example (padded to a
    shared max) rather than being fixed across the batch.
    """
    NUM_TOKENS = 3          # informational only; forward accepts any T
    TOKEN_DIM = 1024

    def __init__(self, token_dim=TOKEN_DIM, hidden=768, num_soc_classes=27,
                 dropout=0.2, prior_logits=None, num_heads=4, init_weight_scale=0.01):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden = int(hidden)
        self.num_soc_classes = int(num_soc_classes)
        self.num_heads = int(num_heads)

        # shared per-token encoder
        self.token_encoder = nn.Sequential(
            nn.Linear(self.token_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # one learnable query per SOC label
        self.label_queries = nn.Parameter(torch.randn(num_soc_classes, hidden) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        # group-wise head: each label projects ITS OWN vector to a scalar logit
        self.group_weight = nn.Parameter(torch.empty(num_soc_classes, hidden))
        self.group_bias = nn.Parameter(torch.zeros(num_soc_classes))
        self._reset_parameters(prior_logits, init_weight_scale)

    def _reset_parameters(self, prior_logits=None, init_weight_scale=0.01):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # small last-layer weights -> logits start at the prior, curbing early over-prediction (#5)
        nn.init.xavier_uniform_(self.group_weight)
        self.group_weight.data.mul_(init_weight_scale)
        if prior_logits is not None:
            with torch.no_grad():
                self.group_bias.copy_(prior_logits.to(self.group_bias.device))

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Accept (B, T, token_dim) or flat (B, T*token_dim). Keeps ALL tokens (no count truncation)."""
        if x.dim() == 2:
            x = x.view(x.size(0), -1, self.token_dim)
        if x.size(-1) > self.token_dim:
            x = x[..., : self.token_dim]
        elif x.size(-1) < self.token_dim:
            raise ValueError(f"Expected last dim >= {self.token_dim}, got {x.size(-1)}")
        return x

    def forward(self, x: torch.Tensor, return_attn: bool = False, key_padding_mask: Optional[torch.Tensor] = None):
        """key_padding_mask: optional (B, T) bool tensor, True at padded token positions.
        Needed once T varies per example (e.g. a variable count of graph-retrieval
        neighbors on top of the fixed BM25/dense ones) -- pad every example's token set to
        a shared T so they can still be batched, and mask the padding out of attention."""
        x = self._prepare_tokens(x)                       # (B, T, token_dim)
        h = self.token_encoder(x)                         # (B, T, hidden)
        q = self.label_queries.unsqueeze(0).expand(h.size(0), -1, -1)  # (B, L, hidden)
        attn_out, attn_w = self.cross_attn(q, h, h, key_padding_mask=key_padding_mask, need_weights=return_attn)
        z = self.attn_norm(q + attn_out)                  # (B, L, hidden)
        z = self.ffn_norm(z + self.ffn(z))                # (B, L, hidden)
        logits = (z * self.group_weight.unsqueeze(0)).sum(-1) + self.group_bias  # (B, L)
        if return_attn:
            return logits, attn_w
        return logits


def print_soc_label_query_model_structure(
    model: Optional[SOCLabelQueryModel] = None,
    batch_size: int = 2,
    num_tokens: int = 10,
    **model_kwargs,
) -> SOCLabelQueryModel:
    """Print the SOCLabelQueryModel layer-by-layer structure (output shapes + param counts)
    for a forward pass on a dummy ``(batch_size, num_tokens, token_dim)`` input, via torchinfo
    (``pip install torchinfo``).

    Pass an existing ``model`` to inspect it as-is, or omit it to build a fresh one from
    ``model_kwargs``. Returns the model so it can be reused.
    """
    from torchinfo import summary

    if model is None:
        model = SOCLabelQueryModel(**model_kwargs)

    device = next(model.parameters()).device
    dummy = torch.randn(batch_size, num_tokens, model.token_dim, device=device)
    summary(model, input_data=dummy)

    return model


def visualize_soc_label_query_model(
    model: Optional[SOCLabelQueryModel] = None,
    batch_size: int = 2,
    num_tokens: int = 10,
    save_path: Optional[Path] = None,
    **model_kwargs,
):
    """Render the SOCLabelQueryModel computation graph via torchviz (``pip install torchviz``;
    also needs the system ``graphviz`` binary, i.e. ``dot``).

    Traces one forward pass on a random dummy input of shape
    ``(batch_size, num_tokens, token_dim)`` and returns the resulting ``graphviz.Digraph``.
    Pass an existing ``model`` to visualize it as-is, or omit it to build a fresh one from
    ``model_kwargs``. If ``save_path`` is given (no extension), renders a PNG there.
    """
    from torchviz import make_dot

    if model is None:
        model = SOCLabelQueryModel(**model_kwargs)
    model.eval()

    device = next(model.parameters()).device
    dummy = torch.randn(batch_size, num_tokens, model.token_dim, device=device, requires_grad=True)
    logits = model(dummy)

    dot = make_dot(logits, params=dict(model.named_parameters()))
    dot.attr(rankdir="TB")

    if save_path:
        dot.render(str(save_path), format="png", cleanup=True)
        print(f"Saved graph to {save_path}.png")

    return dot


def plot_soc_label_query_model_diagram(
    model: Optional[SOCLabelQueryModel] = None,
    save_path: Optional[Path] = None,
    **model_kwargs,
):
    """Draw a simple module-level block diagram of the SOCLabelQueryModel architecture
    (token encoder -> cross-attention -> FFN -> group-wise head). Much coarser than
    :func:`visualize_soc_label_query_model`'s full autograd graph -- one box per logical
    stage, matching the steps in ``SOCLabelQueryModel.forward``.

    Pass an existing ``model`` to label shapes from it, or omit it to build a fresh one
    from ``model_kwargs`` purely to read off dimensions. Returns a ``graphviz.Digraph``;
    renders a PNG to ``save_path`` (no extension) if given.
    """
    from graphviz import Digraph

    if model is None:
        model = SOCLabelQueryModel(**model_kwargs)

    token_dim, hidden, num_soc = model.token_dim, model.hidden, model.num_soc_classes

    dot = Digraph()
    dot.attr(rankdir="TB")
    dot.attr("node", shape="box", style="filled", fillcolor="#EAF2FB", fontname="Helvetica")

    dot.node("tokens", f"Retrieved tokens\n(T, {token_dim})\npadded to shared T across batch")
    dot.node("mask", f"key_padding_mask\n(T,) bool, True = pad", fillcolor="#F5F5F5", shape="note")
    dot.node("queries", f"Label queries\n({num_soc}, {hidden})", fillcolor="#FBEAEA")
    dot.node("encoder", f"Token encoder\nLinear -> LayerNorm -> GELU -> Dropout\n(T, {hidden})")
    dot.node("attn", f"Cross-attention\nqueries attend over tokens\n(padding masked out)\n({num_soc}, {hidden})")
    dot.node("attn_norm", f"Add & LayerNorm\n({num_soc}, {hidden})")
    dot.node("ffn", f"FFN\nLinear -> GELU -> Dropout -> Linear\n({num_soc}, {hidden})")
    dot.node("ffn_norm", f"Add & LayerNorm\n({num_soc}, {hidden})")
    dot.node("head", f"Group-wise head\nper-label weight . vector + bias\n({num_soc},)", fillcolor="#EAFBEA")

    dot.edge("tokens", "encoder")
    dot.edge("encoder", "attn")
    dot.edge("mask", "attn", style="dashed")
    dot.edge("queries", "attn")
    dot.edge("queries", "attn_norm", label="residual")
    dot.edge("attn", "attn_norm")
    dot.edge("attn_norm", "ffn")
    dot.edge("attn_norm", "ffn_norm", label="residual")
    dot.edge("ffn", "ffn_norm")
    dot.edge("ffn_norm", "head")

    if save_path:
        dot.render(str(save_path), format="png", cleanup=True)
        print(f"Saved diagram to {save_path}.png")

    return dot


class FocalMultiLabelLoss(nn.Module):
    """
    Symmetric focal loss for multi-label BCE (Lin et al., 2017 style).
    Uses the same per-class pos_weight as weighted BCE when provided.
    """

    def __init__(self, gamma: float = 1.5, pos_weight: Optional[torch.Tensor] = None, 
                 eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pw = getattr(self, "pos_weight", None)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal = (1.0 - p_t.clamp(min=self.eps)) ** self.gamma
        return (focal * bce).mean()


def _canonical_loss_name(loss_name: str) -> str:
    """Normalize free-form loss labels (e.g. 'focal(γ=1.5)') to a canonical key."""
    name = loss_name.lower().strip()
    # Strip any parenthetical annotation like 'focal(γ=1.5)' -> 'focal'.
    name = name.split("(")[0].strip()
    if name in ("softmax", "bce_unweighted", "bce_flat"):
        return "softmax"
    if name in ("bce", "weighted_bce", "bce_pos_weight", "wtd_bce"):
        return "wtd_bce"
    if name in ("focal", "focal_loss"):
        return "focal"
    if name in ("zero_inflated", "asl", "zil", "zero_inflated_multilabel"):
        return "zero_inflated"
    if name in ("zero_inflated_la", "zil_la", "logit_adjusted", "logit_adjustment", "la"):
        return "zero_inflated_la"
    raise ValueError(
        f"Unknown loss_name={loss_name!r}. Use softmax | wtd_bce | focal | zero_inflated | zero_inflated_la."
    )


def make_criterion(
    loss_name: str,
    pos_weight: torch.Tensor,
    zero_inflated_cls: type,
    *,
    focal_gamma: float = 2.0,
    gamma_neg: int = 2,
    pos_weight_blend: float = 1.0,
    fp_penalty_weight: float = 0.2,
    fp_alpha: int = 6,
    pos_rate: Optional[torch.Tensor] = None,
    tau: float = 1.0,
    ) -> nn.Module:
    """Build criterion by name. `zero_inflated_cls` is the notebook's ZeroInflatedMultiLabelLoss."""
    name = _canonical_loss_name(loss_name)
    if name == "softmax":
        # Simple unweighted BCE baseline (like softmax for multi-label)
        return nn.BCEWithLogitsLoss(pos_weight=None)
    if name == "wtd_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalMultiLabelLoss(gamma=focal_gamma, pos_weight=pos_weight)
    if name == "zero_inflated":
        return zero_inflated_cls(
            gamma_neg=gamma_neg,
            pos_weight=pos_weight,
            fp_penalty_weight=fp_penalty_weight,
            fp_alpha=fp_alpha,
            pos_weight_blend=pos_weight_blend,
        )
    if name == "zero_inflated_la":
        # Logit-adjusted variant: pos_rate/tau replace pos_weight as the
        # imbalance correction (see ZeroInflatedMultiLabelLoss docstring), so
        # pos_weight is left off to avoid stacking both corrections. Keep
        # pos_weight_blend as passed through -- it controls the ASL/BCE mix,
        # not pos_weight, and the tuned value (e.g. 1.0 = pure BCE) is what
        # makes this comparable to the "zero_inflated" run.
        if pos_rate is None:
            raise ValueError("loss_name='zero_inflated_la' requires pos_rate (per-class positive rate).")
        return zero_inflated_cls(
            gamma_neg=gamma_neg,
            pos_weight=None,
            fp_penalty_weight=fp_penalty_weight,
            fp_alpha=fp_alpha,
            pos_weight_blend=pos_weight_blend,
            pos_rate=pos_rate,
            tau=tau,
        )
    raise ValueError(f"Unknown loss_name={loss_name!r}. Use softmax | wtd_bce | focal | zero_inflated | zero_inflated_la.")


LOSS_DISPLAY_NAMES = {
    "softmax": "Softmax/BCE (unweighted)",
    "wtd_bce": "BCE (pos_weight)",
    "focal": "Focal (γ=2, pos_weight)",
    "zero_inflated": "ZeroInflated/ASL",
    "zero_inflated_la": "ZeroInflated + Logit-Adj",
}

# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_result_from_mapping(data: Dict[str, object]) -> "RunResult":
    return RunResult(**data)


@dataclass
class RunResult:
    loss_name: str
    seed: int
    best_epoch: int
    best_val_macro_f1: float
    val_loss: float
    val_macro_p: float
    val_macro_r: float
    val_macro_f1: float
    val_macro_auroc: float
    val_macro_auprc: float
    val_micro_p: float
    val_micro_r: float
    val_micro_f1: float
    val_micro_auroc: float
    val_micro_auprc: float
    val_threshold: float
    val_pred_density: float
    train_seconds: float

    def __reduce__(self):
        return _run_result_from_mapping, (asdict(self),)


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return prec, rec, f1, float(y_pred.mean())


def _micro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return prec, rec, f1


def micro_average_precision(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Threshold-free micro-averaged AUPRC for multilabel outputs."""
    return float(average_precision_score(np.asarray(y_true), np.asarray(probs), average="micro"))


# NOTE: macro_average_precision is defined once, further below (near the other
# threshold/decoding helpers) — do not redefine it here.


def macro_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Threshold-free macro-averaged AUROC, skipping degenerate classes (only one label present)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    aucs = [
        roc_auc_score(y_true[:, j], probs[:, j])
        for j in range(y_true.shape[1])
        if y_true[:, j].min() != y_true[:, j].max()  # both classes present
    ]
    return float(np.mean(aucs)) if aucs else float("nan")


def micro_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Threshold-free micro-averaged AUROC; NaN if the pooled labels are single-class."""
    try:
        return float(roc_auc_score(np.asarray(y_true), np.asarray(probs), average="micro"))
    except ValueError:
        return float("nan")


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-class precision, recall, F1, and positive support."""
    n_classes = y_true.shape[1]
    return pd.DataFrame(
        {
            "class_idx": np.arange(n_classes),
            "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=None, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=None, zero_division=0),
            "support": y_true.sum(axis=0),
        }
    )


# ---------------------------------------------------------------------------
# Decoding / thresholding (target = macro-F1)
#   3. macro-AUPRC  : threshold-free ceiling diagnostic
#   1. per-class F1 : one threshold per SOC, tuned on val
#   2. top-k        : cardinality-aware decode to fight over-prediction
# All operate on numpy (y_true, probs); tune on val, report on a held-out set.
# ---------------------------------------------------------------------------

def macro_average_precision(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Threshold-free macro AUPRC (mean average precision over classes with >=1 positive).

    Use as the fork-in-the-road diagnostic: if this is decent the model can separate
    the classes and the macro-F1 problem is purely the decoder; if it is ~0.4 too,
    the ceiling is the model/features, not the threshold.
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    has_pos = y_true.sum(axis=0) > 0
    if has_pos.sum() == 0:
        return float("nan")
    ap = average_precision_score(
        y_true[:, has_pos], probs[:, has_pos], average=None
    )
    return float(np.mean(np.atleast_1d(ap)))


def per_class_average_precision(
    y_true: np.ndarray, probs: np.ndarray
) -> np.ndarray:
    """Per-class average precision (AUPRC); NaN for classes with no positives."""
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n_classes = y_true.shape[1]
    ap = np.full(n_classes, np.nan, dtype=float)
    has_pos = np.where(y_true.sum(axis=0) > 0)[0]
    for c in has_pos:
        ap[c] = average_precision_score(y_true[:, c], probs[:, c])
    return ap


def tune_global_threshold_f1(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> float:
    """Single threshold (shared by all classes) that maximizes macro-F1 on this set."""
    grid = np.arange(0.05, 0.95, 0.01) if thresholds is None else np.asarray(thresholds)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (probs >= t).astype(float), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def tune_per_class_thresholds_f1(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    *,
    no_pos_threshold: float = 1.0,
) -> np.ndarray:
    """
    One threshold per class, each chosen to maximize *that class's* F1 on this set.

    This is the decoder that matches a macro-F1 objective: rare SOCs get high
    thresholds (stop firing), frequent SOCs get low ones. Tune on val, apply to test.
    Classes with no positive support here get `no_pos_threshold` (default 1.0 = never fire).
    """
    grid = np.arange(0.05, 0.95, 0.01) if thresholds is None else np.asarray(thresholds)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n_classes = y_true.shape[1]
    out = np.full(n_classes, 0.5, dtype=float)
    for c in range(n_classes):
        yt = y_true[:, c]
        if yt.sum() == 0:
            out[c] = no_pos_threshold
            continue
        pc = probs[:, c]
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(yt, (pc >= t).astype(float), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        out[c] = best_t
    return out


def apply_per_class_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Binarize probs with a per-class threshold vector."""
    return (np.asarray(probs) >= np.asarray(thresholds)[None, :]).astype(float)


# ---------------------------------------------------------------------------
# kNN label-vote baseline: evaluate + compare variants (mirrors the
# RunResult / summarize_comparison / plot_loss_comparison / summary_comparison_experiment
# pattern used for the trained-model loss comparison, so both are readable side by side).
# ---------------------------------------------------------------------------

@dataclass
class KnnVoteResult:
    variant_name: str
    threshold_mode: str
    val_macro_p: float
    val_macro_r: float
    val_macro_f1: float
    val_macro_auroc: float
    val_macro_auprc: float
    val_pred_density: float


def evaluate_knn_vote(
    y_true: np.ndarray,
    prior: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    variant_name: str,
    *,
    threshold_mode: str = "f1_per_class",
    thresholds: Optional[np.ndarray] = None,
) -> KnnVoteResult:
    """
    Tune a decode threshold for one kNN vote prior on train_idx, score on test_idx.

    Mirrors train_one_loss's threshold_mode dispatch so the kNN baseline and the
    trained models are decoded with the same convention:
      - "f1_per_class" (default): one threshold per SOC maximizing that class's own
        F1 on train_idx (`tune_per_class_thresholds_f1`) — the fix for a single
        global cutoff over- or under-firing on rare vs. frequent SOCs.
      - "f1_global": single shared threshold maximizing macro-F1 (`tune_global_threshold_f1`).
    AUROC/AUPRC are threshold-free ranking diagnostics computed directly on test_idx.
    """
    y_true = np.asarray(y_true)
    prior = np.asarray(prior)

    if threshold_mode == "f1_per_class":
        thr = tune_per_class_thresholds_f1(y_true[train_idx], prior[train_idx], thresholds=thresholds)
        y_pred = apply_per_class_thresholds(prior[test_idx], thr)
    elif threshold_mode == "f1_global":
        thr = tune_global_threshold_f1(y_true[train_idx], prior[train_idx], thresholds=thresholds)
        y_pred = (prior[test_idx] >= thr).astype(float)
    else:
        raise ValueError(
            f"Unknown threshold_mode={threshold_mode!r}. Use 'f1_per_class' | 'f1_global'."
        )

    p, r, f1, dens = _macro_metrics(y_true[test_idx], y_pred)
    auroc = macro_roc_auc(y_true[test_idx], prior[test_idx])
    auprc = macro_average_precision(y_true[test_idx], prior[test_idx])
    return KnnVoteResult(
        variant_name=variant_name,
        threshold_mode=threshold_mode,
        val_macro_p=p,
        val_macro_r=r,
        val_macro_f1=f1,
        val_macro_auroc=auroc,
        val_macro_auprc=auprc,
        val_pred_density=dens,
    )


def summarize_knn_vote_results(results: Sequence[KnnVoteResult]) -> pd.DataFrame:
    """One row per variant, sorted by macro-F1 (no seeds/std — kNN vote is deterministic)."""
    df = pd.DataFrame([asdict(r) for r in results])
    return df.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)


_KNN_VOTE_DEFAULT_VARIANTS = ("all 10 mean", "dense mean", "bm25 mean")

_KNN_VOTE_LABEL_MAP = {
    "all 10 mean": "Joint Retrievals",
    "dense mean": "Dense Retrievals",
    "bm25 mean": "BM25 Retrievals",
}


def _aggregate_knn_vote_variants(
    df: pd.DataFrame, variants: Sequence[str] = _KNN_VOTE_DEFAULT_VARIANTS
) -> pd.DataFrame:
    """Filter to the requested variant prefixes and mean/std-aggregate by variant.

    Matches by prefix (e.g. "dense mean" matches "dense mean (5)") so callers don't
    have to know the exact suffix. Std is 0.0 when a variant has a single row
    (kNN vote is deterministic unless evaluated across multiple seeds/folds).
    """
    group = df["variant_name"].map(
        lambda name: next((p for p in variants if name.startswith(p)), None)
    )
    filtered = df.assign(_group=group).dropna(subset=["_group"])
    agg = (
        filtered.groupby("_group", sort=False)
        .agg(
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", "std"),
            val_macro_r_mean=("val_macro_r", "mean"),
            val_macro_r_std=("val_macro_r", "std"),
            val_macro_p_mean=("val_macro_p", "mean"),
            val_macro_p_std=("val_macro_p", "std"),
            val_macro_auprc_mean=("val_macro_auprc", "mean"),
            val_macro_auprc_std=("val_macro_auprc", "std"),
        )
        .reindex(variants)
        .dropna(how="all")
        .fillna(0.0)
    )
    return agg


def plot_knn_vote_comparison(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    legend_loc: str = "best",
    title: Optional[str] = "kNN Label-Vote Baseline: Macro-F1 vs Macro-Recall vs Macro-Precision vs Macro-AUPRC",
    variants: Sequence[str] = _KNN_VOTE_DEFAULT_VARIANTS,
    xtick_labels: Optional[Sequence[str]] = None,
) -> None:
    """Grouped bar chart of macro-F1/Recall/Precision/AUPRC, restricted to the unweighted-mean
    variants ("all 10 mean" / "dense mean" / "bm25 mean" by default), mean±std across
    rows per variant, with bold in-bar value labels near the top (mirrors plot_loss_comparison).
    """
    import matplotlib.pyplot as plt

    agg = _aggregate_knn_vote_variants(df, variants)
    if agg.empty:
        print(f"(kNN vote comparison plot skipped: no rows matched variants={variants!r})")
        return
    labels = agg.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.2
    err_kw = dict(capsize=4, ecolor="#333333", elinewidth=1.2)

    bars_f1 = ax.bar(x - 1.5 * width, agg["val_macro_f1_mean"], width=width, yerr=agg["val_macro_f1_std"],
                      color="#4C72B0", label="Macro-F1", error_kw=err_kw)
    bars_r = ax.bar(x - 0.5 * width, agg["val_macro_r_mean"], width=width, yerr=agg["val_macro_r_std"],
                     color="#55A868", label="Macro-Recall", error_kw=err_kw)
    bars_p = ax.bar(x + 0.5 * width, agg["val_macro_p_mean"], width=width, yerr=agg["val_macro_p_std"],
                     color="#C44E52", label="Macro-Precision", error_kw=err_kw)
    bars_pr = ax.bar(x + 1.5 * width, agg["val_macro_auprc_mean"], width=width, yerr=agg["val_macro_auprc_std"],
                      color="#8172B2", label="Macro-AUPRC", error_kw=err_kw)

    ax.set_xticks(x)
    tick_labels = (
        xtick_labels if xtick_labels is not None
        else [_KNN_VOTE_LABEL_MAP.get(name, name) for name in labels]
    )
    ax.set_xticklabels(tick_labels, fontsize=14)
    ax.set_ylabel("Validation score", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    # if title:
    #     ax.set_title(title, fontsize=15)
    tops = (
        (agg["val_macro_f1_mean"] + agg["val_macro_f1_std"]).tolist()
        + (agg["val_macro_r_mean"] + agg["val_macro_r_std"]).tolist()
        + (agg["val_macro_p_mean"] + agg["val_macro_p_std"]).tolist()
        + (agg["val_macro_auprc_mean"] + agg["val_macro_auprc_std"]).tolist()
    )
    ymax = max([0.0, *tops])
    ax.set_ylim(0, min(1.0, ymax * 1.2 if ymax > 0 else 1.0))
    ax.legend(loc=legend_loc, ncols=4, fontsize=12, frameon=False, title=None)

    # Value labels on bars, placed inside the bar near the top edge.
    def _label_bars(container):
        for rect in container:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h * 0.90),
                    ha="center",
                    va="top",
                    fontsize=11,
                    color="black",
                    fontweight="bold",
                    clip_on=True,
                )

    _label_bars(bars_f1)
    _label_bars(bars_r)
    _label_bars(bars_p)
    _label_bars(bars_pr)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.show()


def summary_knn_vote_experiment(
    results: Sequence[KnnVoteResult],
    output_name: str = "knn_vote_comparison.png",
    output_path: Optional[Path] = None,
    verbose: bool = True,
    plot: bool = True,
    legend_loc: str = "best",
    plot_variants: Sequence[str] = _KNN_VOTE_DEFAULT_VARIANTS,
    plot_xtick_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Mirrors summary_comparison_experiment(): print a ranked table, flag the best
    variant, and plot macro-F1/Recall/Precision/AUPRC across kNN label-vote variants
    (plus any extra variants passed in `results`, e.g. a trained ML classifier)."""
    df = summarize_knn_vote_results(results)

    if verbose and not df.empty:
        print("\n" + "=" * 72)
        print("kNN label-vote comparison (test split, threshold-tuned on train)")
        print("=" * 72)
        display_cols = [
            "variant_name",
            "threshold_mode",
            "val_macro_f1",
            "val_macro_r",
            "val_macro_p",
            "val_macro_auroc",
            "val_macro_auprc",
            "val_pred_density",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        best = df.iloc[0]
        print(f"\n✓ Best kNN vote variant by macro-F1: {best['variant_name']} ({best['val_macro_f1']:.4f})")
        if len(df) > 1:
            second = df.iloc[1]
            print(
                f"  F1 advantage over 2nd place ({second['variant_name']}): "
                f"+{best['val_macro_f1'] - second['val_macro_f1']:.4f}"
            )

    if plot and not df.empty:
        try:
            plot_knn_vote_comparison(
                df,
                save_path=(f"{output_path}/{output_name}") if output_path else None,
                legend_loc=legend_loc,
                variants=plot_variants,
                xtick_labels=plot_xtick_labels,
            )
        except Exception as exc:
            print(f"(kNN vote comparison plot skipped: {exc})")

    return df


def expected_cardinality(probs: np.ndarray) -> np.ndarray:
    """Per-row expected number of active labels E[k] = sum(prob), clipped to [1, C]."""
    probs = np.asarray(probs)
    k = np.round(probs.sum(axis=1)).astype(int)
    return np.clip(k, 1, probs.shape[1])


def topk_decode(probs: np.ndarray, k) -> np.ndarray:
    """
    Keep the top-k highest-probability labels per row. `k` is an int or a per-row array.

    Cardinality-aware decoding: caps over-prediction structurally instead of relying
    on a threshold. Pair with `expected_cardinality(probs)` for a data-driven k.
    """
    probs = np.asarray(probs)
    n, C = probs.shape
    k_arr = np.full(n, int(k)) if np.isscalar(k) else np.asarray(k, dtype=int)
    pred = np.zeros_like(probs)
    order = np.argsort(-probs, axis=1)
    for i in range(n):
        ki = int(min(max(k_arr[i], 0), C))
        pred[i, order[i, :ki]] = 1.0
    return pred


def _cap_to_topk(pred: np.ndarray, probs: np.ndarray, k) -> np.ndarray:
    """Among already-positive labels in each row, keep at most the top-k by prob."""
    pred = np.asarray(pred).copy()
    n = pred.shape[0]
    k_arr = np.full(n, int(k)) if np.isscalar(k) else np.asarray(k, dtype=int)
    for i in range(n):
        pos = np.where(pred[i] > 0)[0]
        if len(pos) > k_arr[i]:
            keep = pos[np.argsort(-probs[i, pos])[: int(k_arr[i])]]
            pred[i] = 0.0
            pred[i, keep] = 1.0
    return pred


def compare_decode_strategies(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    y_eval: np.ndarray,
    probs_eval: np.ndarray,
    *,
    recall_floor_threshold: Optional[float] = None,
    thresholds: Optional[np.ndarray] = None,
    label: str = "",
) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """
    Tune every decoder on (y_val, probs_val); report macro metrics on (y_eval, probs_eval).

    Strategies compared (all judged by macro-F1):
      - global F1        : single val-tuned threshold that maxes macro-F1
      - per-class F1     : one val-tuned threshold per SOC  (step 1)
      - top-k E[card]    : keep top-k, k = round(sum probs)   (step 2)
      - per-class ∩ top-k: per-class F1 gated by the cardinality cap
      - recall-floor     : the existing global threshold (baseline), if provided

    Returns (comparison_df, per_class_thresholds, global_threshold).
    """
    y_val, probs_val = np.asarray(y_val), np.asarray(probs_val)
    y_eval, probs_eval = np.asarray(y_eval), np.asarray(probs_eval)

    rows: List[Dict] = []

    def add(name: str, pred_eval: np.ndarray, note: str = "") -> None:
        p, r, f1, d = _macro_metrics(y_eval, pred_eval)
        mp, mr, mf1 = _micro_metrics(y_eval, pred_eval)
        rows.append(
            {
                "strategy": name,
                "macro_p": p,
                "macro_r": r,
                "macro_f1": f1,
                "micro_p": mp,
                "micro_r": mr,
                "micro_f1": mf1,
                "micro_auprc": micro_average_precision(y_eval, probs_eval),
                "pred_density": d,
                "note": note,  # threshold detail kept out of `strategy` so names group cleanly
            }
        )

    gt = tune_global_threshold_f1(y_val, probs_val, thresholds)
    add("global F1", (probs_eval >= gt).astype(float), note=f"t={gt:.2f}")

    pct = tune_per_class_thresholds_f1(y_val, probs_val, thresholds)
    pred_pc = apply_per_class_thresholds(probs_eval, pct)
    add("per-class F1", pred_pc, note=f"t={pct.min():.2f}-{pct.max():.2f}")

    k_eval = expected_cardinality(probs_eval)
    add("top-k E[card]", topk_decode(probs_eval, k_eval), note=f"k~{k_eval.mean():.1f}")

    add("per-class F1 ∩ top-k", _cap_to_topk(pred_pc, probs_eval, k_eval))

    if recall_floor_threshold is not None:
        add(
            "recall-floor",
            (probs_eval >= recall_floor_threshold).astype(float),
            note=f"t={recall_floor_threshold:.2f}",
        )

    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out.insert(0, "eval_set", label)
    return out, pct, gt

def normalizeVect(v):
    """Normalize a vector by min-max."""
    v = np.asarray(v, dtype=np.float32)
    v_min = np.min(v)
    v_max = np.max(v)
    if v_max - v_min < 1e-12:
        return v
    return (v - v_min) / (v_max - v_min)

def _best_f1_over_grid(y_col: np.ndarray, prob_col: np.ndarray, grid: np.ndarray) -> float:
    """Best F1 for one class over a threshold grid (used by per-class alpha search)."""
    best = -1.0
    for t in grid:
        f1 = f1_score(y_col, (prob_col >= t).astype(float), zero_division=0)
        if f1 > best:
            best = f1
    return best


def evaluate_fusion(
    y_val: np.ndarray,
    model_prob_val: np.ndarray,
    knn_prior_val: np.ndarray,
    y_eval: np.ndarray,
    model_prob_eval: np.ndarray,
    knn_prior_eval: np.ndarray,
    *,
    alphas: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    label: str = "",
) -> Tuple[pd.DataFrame, float, np.ndarray]:
    """
    Fuse a kNN label-vote prior with model probabilities (probability space):

        fused = alpha * knn_prior + (1 - alpha) * model_prob

    All quantities must be row-aligned (same query rows across model & prior).
    Everything (alpha, per-class thresholds) is tuned on VAL and reported on EVAL.

    Variants compared (macro-F1):
      - model only              (alpha = 0)
      - kNN only                (alpha = 1)
      - fusion (global alpha)   one alpha for all SOCs, swept on val
      - fusion (per-class alpha) each SOC picks its own alpha on val

    Returns (results_df, best_global_alpha, per_class_alpha).
    """
    yv, mv, kv = map(np.asarray, (y_val, model_prob_val, knn_prior_val))
    ye, me, ke = map(np.asarray, (y_eval, model_prob_eval, knn_prior_eval))
    alphas = np.arange(0.0, 1.0001, 0.1) if alphas is None else np.asarray(alphas)
    grid = np.arange(0.05, 0.95, 0.05) if thresholds is None else np.asarray(thresholds)
    C = yv.shape[1] #27 classes

    rows: List[Dict] = []

    def add(name: str, prior_val: np.ndarray, prior_eval: np.ndarray, alpha_note) -> None:
        thr = tune_per_class_thresholds_f1(yv, prior_val, grid)     # tune decoder on val
        p, r, f1, d = _macro_metrics(ye, apply_per_class_thresholds(prior_eval, thr))
        rows.append(
            {
                "variant": name,
                "alpha": alpha_note,
                "macro_f1": f1,
                "macro_p": p,
                "macro_r": r,
                "auprc": macro_average_precision(ye, prior_eval),
                "density": d,
            }
        )

    add("model only", mv, me, 0.0)
    add("kNN only", kv, ke, 1.0)

    # --- global alpha: pick the alpha whose fused-val macro-F1 is highest ---
    best_a, best_valf1 = 0.0, -1.0
    for a in alphas:
        fv = a * kv + (1.0 - a) * mv
        thr = tune_per_class_thresholds_f1(yv, fv, grid)
        vf1 = f1_score(yv, apply_per_class_thresholds(fv, thr), average="macro", zero_division=0)
        if vf1 > best_valf1:
            best_valf1, best_a = vf1, float(a)
    add(f"fusion (global α)", best_a * kv + (1 - best_a) * mv,
        best_a * ke + (1 - best_a) * me, round(best_a, 2))

    # --- per-class alpha: each SOC picks the alpha maximizing its own val F1 ---
    pc_alpha = np.zeros(C, dtype=float)
    for c in range(C):
        best_ac, best_cf1 = 0.0, -1.0
        for a in alphas:
            fvc = a * kv[:, c] + (1.0 - a) * mv[:, c]
            cf1 = _best_f1_over_grid(yv[:, c], fvc, grid)
            if cf1 > best_cf1:
                best_cf1, best_ac = cf1, float(a)
        pc_alpha[c] = best_ac
    fv_pc = pc_alpha[None, :] * kv + (1 - pc_alpha[None, :]) * mv
    fe_pc = pc_alpha[None, :] * ke + (1 - pc_alpha[None, :]) * me
    add("fusion (per-class α)", fv_pc, fe_pc, f"mean={pc_alpha.mean():.2f}")

    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out.insert(0, "eval_set", label)
    return out, best_a, pc_alpha


def _as_pos_rate_array(pos_rates) -> np.ndarray:
    if isinstance(pos_rates, torch.Tensor):
        return pos_rates.detach().cpu().numpy().astype(float)
    return np.asarray(pos_rates, dtype=float)


def _as_class_names(class_names, n_classes: int) -> List[str]:
    if class_names is None:
        return [f"class_{i}" for i in range(n_classes)]
    if isinstance(class_names, Integral):
        class_count = int(class_names)
        if class_count != n_classes:
            raise ValueError(f"class_names count {class_count} != n_classes {n_classes}")
        return [f"class_{i}" for i in range(n_classes)]

    names = list(class_names)
    if len(names) != n_classes:
        raise ValueError(f"class_names length {len(names)} != n_classes {n_classes}")
    return names


def _assign_frequency_bins(pos_rates: np.ndarray) -> pd.Series:
    """Tertiles by training prevalence: rare / medium / frequent."""
    labels = ["rare", "medium", "frequent"]
    try:
        return pd.qcut(pos_rates, q=3, labels=labels)
    except ValueError:
        # Too many tied pos_rates for qcut; fall back to rank-based bins.
        ranks = pd.Series(pos_rates).rank(method="first")
        return pd.qcut(ranks, q=3, labels=labels)

import torch.nn.functional as F

def compute_soc_priors(y_multihot):
    """Per-class positive rate and logit prior for zero-inflated labels as initials values."""
    y = y_multihot.float()
    pos_rate = (y.sum(dim=0) / y.size(0)).clamp(1e-4, 1.0 - 1e-4)
    prior_logits = torch.log(pos_rate / (1.0 - pos_rate))
    return pos_rate, prior_logits

class ZeroInflatedMultiLabelLoss(nn.Module):
    """
    ASL + PosWeighted BCE + False Positive Penalty + Logit Adjustment

    Designed for sparse multi-label ADR prediction.

    Components
    ----------
    L = ASL
        + lambda_bce * BCE
        + lambda_fp  * FP_penalty

    FP penalty discourages over-confident
    false positive predictions.

    Logit adjustment (Menon et al., 2021, "Long-tail learning via logit
    adjustment") shifts each class's logit by tau * log(pos_rate / (1 -
    pos_rate)) *inside the loss only*, before ASL/BCE/FP-penalty are
    computed. This is Bayes-consistent for balanced/macro metrics, unlike
    pos_weight or the FP penalty, which just reweight the existing loss and
    tend to trade precision for recall on rare classes without moving
    macro-F1. Pass raw (unadjusted) logits at inference/threshold-tuning
    time as usual -- the adjustment only changes the training gradient.
    If both pos_rate/tau and pos_weight are set they stack; pass
    pos_weight=None when tau > 0 since they target the same imbalance.
    pos_weight_blend is a separate knob (the ASL/BCE mix) and does not
    disable pos_weight on its own -- pos_weight still multiplies the ASL
    positive term even at pos_weight_blend=1.
    """

    def __init__(
        self,
        gamma_neg=2.0,
        # gamma_pos=0.0,
        # clip=0.01,
        eps=1e-8,
        pos_weight=None,
        pos_weight_blend=1,
        fp_penalty_weight=0.2,
        fp_alpha=6,
        pos_rate=None,
        tau=1.0,
    ):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.eps = eps

        self.pos_weight_blend = pos_weight_blend

        self.fp_penalty_weight = fp_penalty_weight
        self.fp_alpha = fp_alpha

        self.tau = tau

        if pos_weight is not None:
            pos_weight = torch.as_tensor(
                pos_weight,
                dtype=torch.float32
            )

            self.register_buffer(
                "pos_weight",
                pos_weight
            )

        if pos_rate is not None:
            pos_rate = torch.as_tensor(
                pos_rate,
                dtype=torch.float32
            ).clamp(eps, 1.0 - eps)

            logit_prior = torch.log(pos_rate / (1.0 - pos_rate))

            self.register_buffer(
                "logit_prior",
                logit_prior
            )

    def forward(
        self,
        logits,
        targets
    ):

        targets = targets.float()

        # ----------------------------------
        # logit adjustment (train-time only;
        # feed raw logits at inference)
        # ----------------------------------

        if hasattr(self, "logit_prior") and self.tau != 0:
            logits = logits + self.tau * self.logit_prior.view(1, -1)

        # ----------------------------------
        # probabilities
        # ----------------------------------

        p = torch.sigmoid(logits)

        p = p.clamp(
            self.eps,
            1.0 - self.eps
        )

        p_neg = 1.0 - p

        # if self.clip > 0:
        #     p_neg = (
        #         p_neg + self.clip
        #     ).clamp(max=1.0)

        # ----------------------------------
        # ASL Positive
        # ----------------------------------
        # gamma_pos = 0 means no modulating factor on positive samples, which is common for zero-inflated data;
        pos_loss = (
            targets
            * (1.0 - p)
            * torch.log(p)
        )

        if hasattr(self, "pos_weight"):
            pos_loss = (
                pos_loss
                * self.pos_weight.view(1, -1)
            )

        # ----------------------------------
        # ASL Negative
        # ----------------------------------

        neg_loss = (
            (1.0 - targets)
            * torch.pow(
                p_neg,
                self.gamma_neg
            )
            * torch.log(p_neg)
        )

        asl_loss = -(pos_loss + neg_loss).mean()

        total_loss = asl_loss

        # ----------------------------------
        # BCE Blend
        # ----------------------------------

        if self.pos_weight_blend > 0:

            bce_loss = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    pos_weight=getattr(self, "pos_weight", None),
                    reduction="mean"
                )
            )

            total_loss = (
                (1.0 - self.pos_weight_blend)
                * total_loss
                +
                self.pos_weight_blend
                * bce_loss
            )

        else:
            bce_loss = torch.tensor(
                0.0,
                device=logits.device
            )

        # ----------------------------------
        # False Positive Penalty
        # ----------------------------------
        #
        # penalize:
        #
        # y = 0
        # p < 0.3   => no penalty
        # p > 0.3   => increasing penalty
        #
        # (1-y) * p^alpha
        # This directly attacks excessive predictions.
        # ----------------------------------
        fp_penalty = ((1-targets)
                        * torch.relu(
                            p - 0.3
                        ).pow(self.fp_alpha)
                    ).mean()

        # fp_penalty = (
        #     (1.0 - targets)
        #     *
        #     torch.pow(
        #         p,
        #         self.fp_alpha
        #     )
        # ).mean()

        total_loss = (
            total_loss
            +
            self.fp_penalty_weight
            * fp_penalty
        )

        metrics = {
            "loss": total_loss.detach(),
            "asl_loss": asl_loss.detach(),
            "bce_loss": bce_loss.detach(),
            "fp_penalty": fp_penalty.detach(),            
        }

        return total_loss #, metrics

def collect_per_class_f1_long(
    artifacts: Dict[str, Dict],
    loader,
    collect_multihot_probs: Callable,
    pos_rates,
    class_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Long-format per-class metrics for each trained loss run.
    Uses each run's validation-tuned threshold from `artifacts[loss][seed]['result']`.
    """
    pos_rates = _as_pos_rate_array(pos_rates)
    rows: List[Dict] = []

    for loss_name, by_seed in artifacts.items():
        for seed, pack in by_seed.items():
            model = pack["model"]
            threshold = pack["result"].val_threshold
            y_true, probs = collect_multihot_probs(model, loader)
            y_pred = (probs >= threshold).astype(float)
            cls_df = per_class_metrics(y_true, y_pred)
            names = _as_class_names(class_names, cls_df.shape[0])

            for _, row in cls_df.iterrows():
                idx = int(row["class_idx"])
                rows.append(
                    {
                        "loss_name": loss_name,
                        "loss_label": LOSS_DISPLAY_NAMES.get(loss_name, loss_name),
                        "seed": seed,
                        "class_idx": idx,
                        "class_name": names[idx],
                        "pos_rate": float(pos_rates[idx]),
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                        "f1": float(row["f1"]),
                        "support": float(row["support"]),
                        "threshold": float(threshold),
                    }
                )

    out = pd.DataFrame(rows)
    if len(out):
        cls_meta = out.groupby("class_idx", as_index=False)["pos_rate"].first()
        cls_meta["freq_bin"] = _assign_frequency_bins(cls_meta["pos_rate"].values)
        out = out.merge(cls_meta[["class_idx", "freq_bin"]], on="class_idx", how="left")
    return out


def build_per_class_f1_wide(
    per_class_long: pd.DataFrame,
    baseline_loss: str = "bce",
    target_loss: str = "zero_inflated",
) -> pd.DataFrame:
    """
    Wide table: one row per class, F1 column per loss, plus delta target - baseline.
    Averages over seeds when multiple runs exist.
    """
    if per_class_long.empty:
        return pd.DataFrame()

    avg = (
        per_class_long.groupby(["class_idx", "class_name", "pos_rate", "freq_bin", "loss_name"], as_index=False)
        .agg(f1=("f1", "mean"), precision=("precision", "mean"), recall=("recall", "mean"), support=("support", "mean"))
    )
    f1_wide = avg.pivot_table(
        index=["class_idx", "class_name", "pos_rate", "freq_bin"],
        columns="loss_name",
        values="f1",
        aggfunc="first",
    ).reset_index()
    f1_wide.columns.name = None

    if baseline_loss in f1_wide.columns:
        f1_wide[f"delta_{target_loss}_minus_{baseline_loss}"] = (
            f1_wide.get(target_loss, np.nan) - f1_wide[baseline_loss]
        )
    return f1_wide.sort_values("pos_rate")


def build_per_class_recall_wide(
    per_class_long: pd.DataFrame,
    baseline_loss: str = "bce",
    target_loss: str = "zero_inflated",
) -> pd.DataFrame:
    """
    Wide table focused on recall: one row per class, recall column per loss, plus delta target - baseline.
    Averages over seeds when multiple runs exist.
    """
    if per_class_long.empty:
        return pd.DataFrame()

    avg = (
        per_class_long.groupby(["class_idx", "class_name", "pos_rate", "freq_bin", "loss_name"], as_index=False)
        .agg(recall=("recall", "mean"), precision=("precision", "mean"), f1=("f1", "mean"), support=("support", "mean"))
    )
    recall_wide = avg.pivot_table(
        index=["class_idx", "class_name", "pos_rate", "freq_bin"],
        columns="loss_name",
        values="recall",
        aggfunc="first",
    ).reset_index()
    recall_wide.columns.name = None

    if baseline_loss in recall_wide.columns:
        recall_wide[f"delta_recall_{target_loss}_minus_{baseline_loss}"] = (
            recall_wide.get(target_loss, np.nan) - recall_wide[baseline_loss]
        )
    return recall_wide.sort_values("pos_rate")


def summarize_f1_by_frequency_bin(per_class_long: pd.DataFrame) -> pd.DataFrame:
    """Mean per-class F1 by frequency bin and loss (averaged over classes in bin)."""
    if per_class_long.empty:
        return pd.DataFrame()

    avg = (
        per_class_long.groupby(["freq_bin", "loss_name", "loss_label"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            n_classes=("class_idx", "nunique"),
            mean_pos_rate=("pos_rate", "mean"),
        )
        .sort_values(["freq_bin", "loss_name"])
    )
    return avg


def summarize_recall_by_frequency_bin(per_class_long: pd.DataFrame) -> pd.DataFrame:
    """Mean per-class Recall by frequency bin and loss (averaged over classes in bin)."""
    if per_class_long.empty:
        return pd.DataFrame()

    avg = (
        per_class_long.groupby(["freq_bin", "loss_name", "loss_label"], as_index=False)
        .agg(
            mean_recall=("recall", "mean"),
            median_recall=("recall", "median"),
            std_recall=("recall", "std"),
            n_classes=("class_idx", "nunique"),
            mean_pos_rate=("pos_rate", "mean"),
        )
        .sort_values(["freq_bin", "loss_name"])
    )
    return avg

def summarize_recall_delta_by_frequency_bin(
    per_class_wide: pd.DataFrame,
    baseline_loss: str = "bce",
    target_loss: str = "zero_inflated",
) -> pd.DataFrame:
    """Mean Recall gain of target vs baseline within each frequency bin."""
    delta_col = f"delta_recall_{target_loss}_minus_{baseline_loss}"
    if per_class_wide.empty or delta_col not in per_class_wide.columns:
        return pd.DataFrame()

    rows = []
    for freq_bin, grp in per_class_wide.groupby("freq_bin", observed=False):
        rows.append(
            {
                "freq_bin": freq_bin,
                "n_classes": len(grp),
                "mean_pos_rate": float(grp["pos_rate"].mean()),
                f"mean_delta_recall_{target_loss}_minus_{baseline_loss}": float(grp[delta_col].mean()),
                "n_improved": int((grp[delta_col] > 1e-4).sum()),
            }
        )
    return pd.DataFrame(rows)

def summarize_f1_delta_by_frequency_bin(
    per_class_wide: pd.DataFrame,
    baseline_loss: str = "bce",
    target_loss: str = "zero_inflated",
) -> pd.DataFrame:
    """Mean F1 gain of target vs baseline within each frequency bin."""
    delta_col = f"delta_{target_loss}_minus_{baseline_loss}"
    if per_class_wide.empty or delta_col not in per_class_wide.columns:
        return pd.DataFrame()

    rows = []
    for freq_bin, grp in per_class_wide.groupby("freq_bin", observed=False):
        rows.append(
            {
                "freq_bin": freq_bin,
                "n_classes": len(grp),
                "mean_pos_rate": grp["pos_rate"].mean(),
                f"mean_f1_{baseline_loss}": grp.get(baseline_loss, pd.Series(dtype=float)).mean(),
                f"mean_f1_{target_loss}": grp.get(target_loss, pd.Series(dtype=float)).mean(),
                f"mean_delta_{target_loss}_minus_{baseline_loss}": grp[delta_col].mean(),
                f"median_delta_{target_loss}_minus_{baseline_loss}": grp[delta_col].median(),
                f"n_improved": int((grp[delta_col] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_f1_by_frequency_bin(
    bin_summary: pd.DataFrame,
    save_path: Optional[Path] = None,
    #title: Optional[str] = "Per-class F1 by SOC frequency bin",
) -> None:
    """Grouped bar chart: mean per-class F1 by frequency bin and loss.

    Pass ``title=None`` to omit the figure title.
    """
    import matplotlib.pyplot as plt

    if bin_summary.empty:
        print("(plot skipped: empty bin summary)")
        return

    order = ["rare", "medium", "frequent"]
    losses = bin_summary["loss_label"].unique().tolist()
    bins_present = [b for b in order if b in set(bin_summary["freq_bin"])]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(bins_present))
    width = 0.8 / max(len(losses), 1)
    # Assign colors by position so any set/labeling of losses renders correctly.
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

    for i, loss_label in enumerate(losses):
        sub = bin_summary[bin_summary["loss_label"] == loss_label].set_index("freq_bin")
        vals = [sub.loc[b, "mean_f1"] if b in sub.index else np.nan for b in bins_present]
        offset = (i - (len(losses) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=loss_label, color=palette[i % len(palette)])

    ax.set_xticks(x)
    ax.set_xticklabels(bins_present)
    ax.set_xlabel("Class frequency bin (by train pos_rate)", fontsize=14)
    ax.set_ylabel("Mean per-class F1", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylim(0, bin_summary["mean_f1"].max() * 1.5)
    # if title:
    #     ax.set_title(title, fontsize=15)
    ax.legend(ncols = 3, loc="best", fontsize=13)

    # Value labels seated in the upper portion of each bar, fully inside it.
    # Anchor the text top at ~90% of the bar height (va="top") so the label
    # scales with the bar and always stays within it, a bit below the top edge.
    for container in ax.containers:
        for rect in container:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h * 0.90),
                    ha="center",
                    va="top",
                    fontsize=11,
                    color="black",
                    fontweight="bold",
                )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.show()


def plot_f1_delta_vs_prevalence(
    per_class_wide: pd.DataFrame,
    baseline_loss: str = "bce",
    target_loss: str = "zero_inflated",
    save_path: Optional[Path] = None,
) -> None:
    """Scatter: pos_rate vs (target F1 - baseline F1). Points below-left = rare classes."""
    import matplotlib.pyplot as plt

    delta_col = f"delta_{target_loss}_minus_{baseline_loss}"
    if per_class_wide.empty or delta_col not in per_class_wide.columns:
        print("(plot skipped: delta column missing)")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        per_class_wide["pos_rate"],
        per_class_wide[delta_col],
        c=per_class_wide["freq_bin"].astype("category").cat.codes,
        cmap="viridis",
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Train pos_rate (class prevalence)")
    ax.set_ylabel(f"Δ F1 ({target_loss} − {baseline_loss})")
    #ax.set_title("ZeroInflated gain vs class rarity")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.show()


def run_per_class_f1_breakdown(
    artifacts: Dict[str, Dict],
    loader,
    collect_multihot_probs: Callable,
    pos_rates,
    class_names: Optional[Sequence[str]] = None,
    *,
    baseline_loss: str = "softmax",
    target_loss: str = "zero_inflated",
    output_dir: Optional[str] = "experiments/outputs",
    plot: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build per-class F1 tables and frequency-bin summaries.

    Returns
    -------
    per_class_long, per_class_wide, bin_summary, delta_by_bin
    """
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    per_class_long = collect_per_class_f1_long(
        artifacts, loader, collect_multihot_probs, pos_rates, class_names=class_names
    )
    per_class_wide = build_per_class_f1_wide(
        per_class_long, baseline_loss=baseline_loss, target_loss=target_loss
    )
    # Recall-focused analysis
    per_class_recall_wide = build_per_class_recall_wide(
        per_class_long, baseline_loss=baseline_loss, target_loss=target_loss
    )
    
    bin_summary = summarize_f1_by_frequency_bin(per_class_long)
    recall_by_bin = summarize_recall_by_frequency_bin(per_class_long)
    delta_by_bin = summarize_f1_delta_by_frequency_bin(
        per_class_wide, baseline_loss=baseline_loss, target_loss=target_loss
    )
    recall_delta_by_bin = summarize_recall_delta_by_frequency_bin(
        per_class_recall_wide, baseline_loss=baseline_loss, target_loss=target_loss
    )

    if verbose and not per_class_wide.empty:
        print("\n" + "=" * 72)
        print("Per-class F1 by Frequency Bin (validation, threshold-tuned)")
        print("=" * 72)
        print(
            bin_summary.pivot(index="freq_bin", columns="loss_label", values="mean_f1")
            .reindex(["rare", "medium", "frequent"])
            .to_string(float_format=lambda x: f"{x:.4f}")
        )
        if not delta_by_bin.empty:
            print(f"\n{target_loss.title()} vs {baseline_loss.title()} — mean ΔF1 by bin:")
            print(
                delta_by_bin[
                    [
                        "freq_bin",
                        "n_classes",
                        "mean_pos_rate",
                        f"mean_delta_{target_loss}_minus_{baseline_loss}",
                        "n_improved",
                    ]
                ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )
        
        print("\n" + "=" * 72)
        print("Per-class Recall by Frequency Bn (validation, threshold-tuned)")
        print("=" * 72)
        print(
            recall_by_bin.pivot(index="freq_bin", columns="loss_label", values="mean_recall")
            .reindex(["rare", "medium", "frequent"])
            .to_string(float_format=lambda x: f"{x:.4f}")
        )
        if not recall_delta_by_bin.empty:
            print(f"\n{target_loss.title()} vs {baseline_loss.title()} — mean Δ Recall by bin:")
            print(
                recall_delta_by_bin[
                    [
                        "freq_bin",
                        "n_classes",
                        "mean_pos_rate",
                        f"mean_delta_recall_{target_loss}_minus_{baseline_loss}",
                        "n_improved",
                    ]
                ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )
        print("\nTop classes by F1 gain (ZeroInflated vs Softmax):")
        delta_col = f"delta_{target_loss}_minus_{baseline_loss}"
        f1_cols = ["class_name", "pos_rate", "freq_bin", baseline_loss, target_loss, delta_col]
        f1_cols = [c for c in f1_cols if c in per_class_wide.columns]
        print(
            per_class_wide.sort_values(delta_col, ascending=False)
            .head(8)[f1_cols]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    if output_path and not per_class_long.empty:
        per_class_long.to_csv(output_path / "loss_comparison_per_class_long.csv", index=False)
        per_class_wide.to_csv(output_path / "loss_comparison_per_class_wide.csv", index=False)
        per_class_recall_wide.to_csv(output_path / "loss_comparison_per_class_recall_wide.csv", index=False)
        bin_summary.to_csv(output_path / "loss_comparison_f1_by_freq_bin.csv", index=False)
        recall_by_bin.to_csv(output_path / "loss_comparison_recall_by_freq_bin.csv", index=False)
        delta_by_bin.to_csv(output_path / "loss_comparison_delta_by_freq_bin.csv", index=False)
        recall_delta_by_bin.to_csv(output_path / "loss_comparison_recall_delta_by_freq_bin.csv", index=False)
        if verbose:
            print(f"\nWrote per-class CSVs to {output_path}/")

    if plot:
        try:
            # F1-focused plots (primary)
            plot_f1_by_frequency_bin(
                bin_summary,
                save_path=(output_path / "loss_comparison_f1_by_freq_bin.png") if output_path else None,
            )
            # Recall plots (secondary)
            plot_recall_by_frequency_bin(
                recall_by_bin,
                save_path=(output_path / "loss_comparison_recall_by_freq_bin.png") if output_path else None,
            )
            plot_f1_delta_vs_prevalence(
                per_class_wide,
                baseline_loss=baseline_loss,
                target_loss=target_loss,
                save_path=(output_path / "loss_comparison_delta_vs_prevalence.png") if output_path else None,
            )
        except Exception as exc:
            print(f"(per-class plot skipped: {exc})")

    return per_class_long, per_class_wide, bin_summary, delta_by_bin


def train_one_loss(
    loss_name: str,
    train_loader,
    val_loader,
    model_factory: Callable[[], nn.Module],
    pos_weight: torch.Tensor,
    device: torch.device,
    zero_inflated_cls: type,
    eval_multihot_loader: Callable,
    collect_multihot_probs: Callable,
    tune_threshold_fn: Optional[Callable] = None,
    *,
    seed: int = 1234,
    num_epochs: int = 30,    
    patience: int = 5,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    threshold: float = 0.2,
    precision_floor: float = 0.35,
    threshold_mode: str = "f1_per_class",
    focal_gamma: float = 2.0,
    gamma_neg: int = 2,
    pos_weight_blend: float = 0.1,
    fp_penalty_weight: float = 0.1,
    fp_alpha: float = 0.1,
    pos_rate: Optional[torch.Tensor] = None,
    tau: float = 1.0,
    verbose: bool = True,
) -> Tuple[nn.Module, RunResult, Dict]:
    """
    Train one model from scratch; early-stop on validation macro-F1.

    threshold_mode controls how validation probabilities are decoded (mis-thresholding
    fix: AUROC/AUPRC can look solid while F1/P/R shows recall >> precision — that means
    the *decoder*, not the model, is off):
      - "f1_per_class" (default): one threshold per SOC class, each chosen to maximize
        that class's own F1 on val (`tune_per_class_thresholds_f1`). Directly optimizes
        the reported macro-F1 instead of satisficing a recall-floor/precision-floor
        constraint, so it stops over-firing once precision starts dropping.
      - "f1_global": single shared threshold maximizing macro-F1 (`tune_global_threshold_f1`).
      - "external": use the caller-supplied ``tune_threshold_fn(y_val, probs_val,
        precision_floor=precision_floor)`` — the old recall-floor-constrained behavior.
        ``tune_threshold_fn`` and ``precision_floor`` are ignored unless this is selected.
    """
    import time

    set_seed(seed)
    model = model_factory().to(device)
    criterion = make_criterion(
        loss_name,
        pos_weight.to(device),
        zero_inflated_cls,
        focal_gamma=focal_gamma,
        gamma_neg=gamma_neg,
        pos_weight_blend=pos_weight_blend,
        fp_penalty_weight=fp_penalty_weight,
        fp_alpha=fp_alpha,
        pos_rate=pos_rate,
        tau=tau,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_val_recall = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict] = []

    t0 = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()
        total_train, n_train = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss for {loss_name} at epoch {epoch + 1}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            bs = x_batch.size(0)
            total_train += loss.item() * bs
            n_train += bs

        tr_loss = total_train / n_train
        va_loss, va_f1, va_rec, pred_rate = eval_multihot_loader(
            model, val_loader, criterion, threshold=threshold
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_macro_f1": va_f1,
            "val_macro_recall": va_rec,
            "pred_density": pred_rate,
        }
        history.append(row)

        if verbose:
            print(
                f"[{LOSS_DISPLAY_NAMES.get(loss_name, loss_name)} | seed={seed}] "
                f"Epoch {epoch + 1}/{num_epochs} | train {tr_loss:.4f} | val {va_loss:.4f} | "
                f"macro-F1 {va_f1:.4f} | macro-R {va_rec:.4f} | pred density {pred_rate:.4f}"
            )

        if va_f1 > best_val_f1 + 1e-4:
            best_val_f1 = va_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch + 1}.")
                break

    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Threshold tuning on validation (same protocol as main notebook)
    val_threshold = threshold
    val_macro_p = val_macro_r = val_macro_f1 = best_val_f1
    val_micro_p = val_micro_r = val_micro_f1 = val_micro_auprc = float("nan")
    val_macro_auroc = val_macro_auprc = val_micro_auroc = float("nan")
    val_pred_density = history[-1]["pred_density"] if history else 0.0
    va_loss = history[-1]["val_loss"] if history else float("nan")

    y_val, probs_val = collect_multihot_probs(model, val_loader)
    per_class_thresholds: Optional[np.ndarray] = None

    if threshold_mode == "f1_per_class":
        # Mis-thresholding fix: one threshold per SOC that maximizes that class's own
        # F1, instead of a single recall-floor-satisficing cutoff that over-fires.
        per_class_thresholds = tune_per_class_thresholds_f1(y_val, probs_val)
        y_pred = apply_per_class_thresholds(probs_val, per_class_thresholds)
        val_threshold = float(np.mean(per_class_thresholds))
    elif threshold_mode == "f1_global":
        val_threshold = tune_global_threshold_f1(y_val, probs_val)
        y_pred = (probs_val >= val_threshold).astype(float)
    elif threshold_mode == "external":
        if tune_threshold_fn is None:
            raise ValueError("threshold_mode='external' requires a tune_threshold_fn")
        val_threshold, _, _, _ = tune_threshold_fn(
            y_val, probs_val, precision_floor=precision_floor
        )
        y_pred = (probs_val >= val_threshold).astype(float)
        _, va_loss, _, _ = eval_multihot_loader(
            model, val_loader, criterion, threshold=val_threshold
        )
    else:
        raise ValueError(
            f"Unknown threshold_mode={threshold_mode!r}. "
            "Use 'f1_per_class' | 'f1_global' | 'external'."
        )

    val_macro_p, val_macro_r, val_macro_f1, val_pred_density = _macro_metrics(y_val, y_pred)
    val_micro_p, val_micro_r, val_micro_f1 = _micro_metrics(y_val, y_pred)
    val_micro_auprc = micro_average_precision(y_val, probs_val)

    # Threshold-free ranking metrics (independent of the decode mode above).
    val_macro_auroc = macro_roc_auc(y_val, probs_val)
    val_macro_auprc = macro_average_precision(y_val, probs_val)
    val_micro_auroc = micro_roc_auc(y_val, probs_val)

    result = RunResult(
        loss_name=loss_name,
        seed=seed,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1,
        val_loss=va_loss,
        val_macro_p=val_macro_p,
        val_macro_r=val_macro_r,
        val_macro_f1=val_macro_f1,
        val_macro_auroc=val_macro_auroc,
        val_macro_auprc=val_macro_auprc,
        val_micro_p=val_micro_p,
        val_micro_r=val_micro_r,
        val_micro_f1=val_micro_f1,
        val_micro_auroc=val_micro_auroc,
        val_micro_auprc=val_micro_auprc,
        val_threshold=val_threshold,
        val_pred_density=val_pred_density,
        train_seconds=train_seconds,
    )
    extra: Dict = {"history": history}
    if per_class_thresholds is not None:
        # Needed to decode any other split (e.g. test/OOT) the same way as val.
        extra["per_class_thresholds"] = per_class_thresholds
    return model, result, extra

def summarize_comparison(df: pd.DataFrame, loss_name: list) -> pd.DataFrame:
    """Aggregate across seeds if multiple runs per loss."""
    agg = (
        df.groupby("loss_name", as_index=False)
        .agg(
            n_runs=("seed", "count"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", "std"),
            val_macro_r_mean=("val_macro_r", "mean"),
            val_macro_r_std=("val_macro_r", "std"),
            val_macro_p_mean=("val_macro_p", "mean"),
            val_macro_p_std=("val_macro_p", "std"),
            val_macro_auroc_mean=("val_macro_auroc", "mean"),
            val_macro_auroc_std=("val_macro_auroc", "std"),
            val_macro_auprc_mean=("val_macro_auprc", "mean"),
            val_macro_auprc_std=("val_macro_auprc", "std"),
            val_micro_f1_mean=("val_micro_f1", "mean"),
            val_micro_f1_std=("val_micro_f1", "std"),
            val_micro_r_mean=("val_micro_r", "mean"),
            val_micro_r_std=("val_micro_r", "std"),
            val_micro_p_mean=("val_micro_p", "mean"),
            val_micro_p_std=("val_micro_p", "std"),
            val_micro_auroc_mean=("val_micro_auroc", "mean"),
            val_micro_auroc_std=("val_micro_auroc", "std"),
            val_micro_auprc_mean=("val_micro_auprc", "mean"),
            val_micro_auprc_std=("val_micro_auprc", "std"),
            val_pred_density_mean=("val_pred_density", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
        )
        .sort_values("val_macro_f1_mean", ascending=False)
    )
    # std is NaN with a single seed; report 0.0 so the column is always numeric.
    for col in (
        "val_macro_f1_std",
        "val_macro_r_std",
        "val_macro_p_std",
        "val_macro_auroc_std",
        "val_macro_auprc_std",
        "val_micro_f1_std",
        "val_micro_r_std",
        "val_micro_p_std",
        "val_micro_auroc_std",
        "val_micro_auprc_std",
    ):
        agg[col] = agg[col].fillna(0.0)
    agg["loss_label"] = agg["loss_name"].map(LOSS_DISPLAY_NAMES).fillna(agg["loss_name"])
    return agg


def plot_loss_comparison(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    legend_loc="best",
    summary: Optional[pd.DataFrame] = None,
    width = 0.3
) -> None:
    """Grouped bar chart of validation macro-F1 / macro-Recall / macro-Precision by loss label.

    Pass a precomputed ``summary`` (from :func:`summarize_comparison`) to avoid recomputing it.
    """
    import matplotlib.pyplot as plt

    if summary is None:
        loss_names = df["loss_name"].unique().tolist()
        summary = summarize_comparison(df, loss_names)
    labels = summary["loss_label"].tolist()
    f1_means = summary["val_macro_f1_mean"].tolist()
    recall_means = summary["val_macro_r_mean"].tolist()
    precision_means = summary["val_macro_p_mean"].tolist()
    # ±1 std across seeds (0.0 when a single seed); shown as error whiskers.
    f1_std = summary.get("val_macro_f1_std", pd.Series(0.0, index=summary.index)).tolist()
    recall_std = summary.get("val_macro_r_std", pd.Series(0.0, index=summary.index)).tolist()
    precision_std = summary.get("val_macro_p_std", pd.Series(0.0, index=summary.index)).tolist()

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))
    width = 0.3
    err_kw = dict(capsize=4, ecolor="#333333", elinewidth=1.2)

    bars_f1 = ax.bar(x - width, f1_means, width=width, yerr=f1_std, color="#4C72B0",
                     label="Macro-F1", error_kw=err_kw)
    bars_r = ax.bar(x, recall_means, width=width, yerr=recall_std, color="#55A868",
                    label="Macro-Recall", error_kw=err_kw)
    bars_p = ax.bar(x + width, precision_means, width=width, yerr=precision_std, color="#C44E52",
                    label="Macro-Precision", error_kw=err_kw)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14) #, rotation=15, ha="right")
    ax.set_ylabel("Validation score", fontsize=14)
    #ax.set_title("Loss Comparison: Macro-F1 vs Macro-Recall vs Macro-Precision", fontsize=15)
    tops = [m + s for m, s in zip(
        (*f1_means, *recall_means, *precision_means),
        (*f1_std, *recall_std, *precision_std),
    )]
    ymax = max([0.0, *tops])
    ax.set_ylim(0, min(1.0, ymax * 1.2 if ymax > 0 else 1.0))
    ax.legend(loc=legend_loc, ncol = 3, fontsize=14, frameon=False, title=None)

    # Value labels on bars, placed inside the bar near the top edge.
    def _label_bars(container):
        for rect in container:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h * 0.90),
                    ha="center",
                    va="top",
                    fontsize=11,
                    color="black",
                    fontweight="bold",
                    clip_on=True,
                )

    _label_bars(bars_f1)
    _label_bars(bars_r)
    _label_bars(bars_p)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.show()


def plot_recall_by_frequency_bin(
    recall_summary: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: Optional[str] = "Recall by Class Frequency & Loss Function",
) -> None:
    """Grouped bar chart of per-class Recall by frequency bin and loss.

    Pass ``title=None`` to omit the figure title.
    """
    import matplotlib.pyplot as plt

    if recall_summary.empty:
        return

    pivot = recall_summary.pivot_table(
        index="freq_bin",
        columns="loss_label",
        values="mean_recall",
        aggfunc="first",
    ).reindex(["rare", "medium", "frequent"])

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, width=0.9, color=["#4C72B0", "#55A868", "#C44E52","#f7072f"], alpha=0.8)
    ax.set_xlabel("Class Frequency Bin", fontsize=14)
    ax.set_ylabel("Mean per-class Recall", fontsize=14)
    ax.tick_params(axis="x", labelrotation=0, labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    # if title:
    #     ax.set_title(title, fontsize=15)
    ax.legend(loc="best", ncols = 3, fontsize=13, frameon=False, title=None)
    ax.set_ylim(0, 1.0)

    # Value labels seated in the upper portion of each bar, fully inside it.
    # Anchor the text top at ~90% of the bar height (va="top") so the label
    # scales with the bar and always stays within it, a bit below the top edge.
    for container in ax.containers:
        for rect in container:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h * 0.90),
                    ha="center",
                    va="top",
                    fontsize=11,
                    color="black",
                    fontweight="bold",
                )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.show()


def run_loss_comparison_experiment(
    train_loader,
    val_loader,
    model_factory: Callable[[], nn.Module],
    pos_weight: torch.Tensor,
    device: torch.device,
    zero_inflated_cls: type,
    eval_multihot_loader: Callable,
    collect_multihot_probs: Callable,
    tune_threshold_fn: Optional[Callable] = None,
    #train_targets: Optional[torch.Tensor] = None,
    *,
    loss_names: Sequence[str] = ("softmax", "zero_inflated"),
    seeds: Sequence[int] = (1234,),
    num_epochs: int = 30,
    patience: int = 5,
    output_dir: Optional[str] = "loss_experiments/outputs",
    save_models: bool = False,
    plot: bool = True,
    verbose: bool = True,    
    gamma_neg: int = 2,
    fp_penalty_weight= 0.05,
    fp_alpha: int = 2,
    pos_weight_blend: float = 0.1000,
    pos_rate: Optional[torch.Tensor] = None,
    tau: float = 1.0,
    **train_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Train one model per loss (and optional multi-seed), return results table + artifacts.

    Returns
    -------
    results_df : one row per (loss_name, seed)
    artifacts : dict[loss_name][seed] -> {model, history, result}
    """
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    rows: List[RunResult] = []
    artifacts: Dict[str, Dict] = {}

    # Normalize free-form labels (e.g. "focal(γ=1.5)") to canonical keys so that
    # criterion lookup, artifacts keys, and display-name mapping all stay in sync.
    loss_names = [_canonical_loss_name(n) for n in loss_names]

    for loss_name in loss_names:
        artifacts[loss_name] = {}
        for seed in seeds:
            if verbose:
                print("\n" + "=" * 72)
                print(f"Training loss={loss_name!r} seed={seed}")
                print("=" * 72)
            model, result, extra = train_one_loss(
                loss_name=loss_name,
                train_loader=train_loader,
                val_loader=val_loader,
                model_factory=model_factory,
                pos_weight=pos_weight,
                device=device,
                zero_inflated_cls=zero_inflated_cls,
                eval_multihot_loader=eval_multihot_loader,
                collect_multihot_probs=collect_multihot_probs,
                tune_threshold_fn=tune_threshold_fn,
                seed=seed,
                num_epochs=num_epochs,
                patience=patience,
                verbose=verbose,              
                gamma_neg=gamma_neg,
                pos_weight_blend=pos_weight_blend,
                fp_penalty_weight=fp_penalty_weight,
                fp_alpha=fp_alpha,
                pos_rate=pos_rate,
                tau=tau,
                **train_kwargs,
            )
            rows.append(result)
            if verbose:
                print(
                    f"  [{loss_name} seed={seed}]\n"
                    f"    macro  F1={result.val_macro_f1:.4f} P={result.val_macro_p:.4f} "
                    f"R={result.val_macro_r:.4f} AUROC={result.val_macro_auroc:.4f} "
                    f"AUPRC={result.val_macro_auprc:.4f}\n"
                    f"    micro  F1={result.val_micro_f1:.4f} P={result.val_micro_p:.4f} "
                    f"R={result.val_micro_r:.4f} AUROC={result.val_micro_auroc:.4f} "
                    f"AUPRC={result.val_micro_auprc:.4f}"
                )
            artifacts[loss_name][seed] = {
                "model": model,
                "history": extra["history"],
                "result": result,
                "per_class_thresholds": extra.get("per_class_thresholds"),
            }
            if save_models and output_path:
                # Differentiate the checkpoint by model type from model_factory:
                # use the factory's name if it is a named function, else fall back
                # to the concrete class it produced (handles lambdas). Prevents
                # different architectures overwriting each other's checkpoints.
                factory_name = getattr(model_factory, "__name__", "")
                model_tag = (
                    factory_name
                    if factory_name and factory_name != "<lambda>"
                    else type(model).__name__
                )
                ckpt = output_path / f"soc_multihot_{model_tag}_{loss_name}_seed{seed}.pth"
                torch.save(model.state_dict(), ckpt)

    results_df = pd.DataFrame([asdict(r) for r in rows])
    results_df["loss_label"] = (
        results_df["loss_name"].map(LOSS_DISPLAY_NAMES).fillna(results_df["loss_name"])
    )

    return results_df, artifacts

def summary_comparison_experiment(
    results_df: pd.DataFrame,
    output_name: str = "loss_comparison_summary.png",
    output_path: Optional[Path] = None,
    verbose: bool = True,
    plot = True,
    legend_loc = "best", 
    width = 0.3   
    ):
    
    loss_names = results_df['loss_name'].unique().tolist()
    summary = summarize_comparison(results_df, loss_names)
    if verbose:
        print("\n" + "=" * 72)
        print("Loss comparison summary (validation, threshold-tuned)")
        print("=" * 72)
        # Sort by F1 — the primary metric this summary reports on.
        summary_f1 = summary.sort_values("val_macro_f1_mean", ascending=False)
        display_cols = [
            "loss_label",
            "n_runs",
            "val_macro_f1_mean",  # F1 first (primary)
            "val_macro_f1_std",
            "val_macro_r_mean",
            "val_macro_p_mean",
            "val_macro_auroc_mean",
            "val_macro_auprc_mean",
            "val_micro_f1_mean",
            "val_micro_r_mean",
            "val_micro_p_mean",
            "val_micro_auroc_mean",
            "val_micro_auprc_mean",
            "val_pred_density_mean",
        ]
        display_cols = [c for c in display_cols if c in summary_f1.columns]
        print(summary_f1[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        best = summary_f1.iloc[0]
        print(
            f"\n✓ Best loss by mean val macro-F1: {best['loss_label']} "
            f"({best['val_macro_f1_mean']:.4f} ± {best.get('val_macro_f1_std', 0.0):.4f})"
        )
        # Show F1 advantage, and flag when it is inside the run-to-run noise band.
        if len(summary_f1) > 1:
            second = summary_f1.iloc[1]
            f1_diff = best["val_macro_f1_mean"] - second["val_macro_f1_mean"]
            pooled_std = max(
                best.get("val_macro_f1_std", 0.0), second.get("val_macro_f1_std", 0.0)
            )
            print(f"  F1 advantage over 2nd place ({second['loss_label']}): +{f1_diff:.4f}")
            if f1_diff < pooled_std:
                print(
                    f"  ⚠ gap ({f1_diff:.4f}) is smaller than the per-loss seed std "
                    f"({pooled_std:.4f}) — treat the ranking as a tie."
                )

    if plot and len(results_df):
        try:
            plot_loss_comparison(
                results_df,
                save_path=(f"{output_path}/{output_name}") if output_path else None,
                legend_loc=legend_loc,
                summary=summary, 
                width = width, # reuse the already-aggregated table
            )
        except Exception as exc:
            print(f"(loss comparison plot skipped: {exc})")

    return summary


def summarize_val_oot_comparison(
    val_df: pd.DataFrame,
    oot_1_df: pd.DataFrame,
    oot_2_df: pd.DataFrame,
    val_cols: Optional[Dict[str, str]] = None,
    oot_1_cols: Optional[Dict[str, str]] = None,
    oot_2_cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Aggregate per-seed macro F1/Recall/Precision into a Validation-vs-OOT summary.

    ``val_df``/``oot_df`` are one-row-per-seed tables (e.g. the per-seed best-checkpoint
    validation rows and ``oot_eval_df``). Column names differ between the two by
    convention (``val_macro_f1`` vs ``macro_f1``); pass ``val_cols``/``oot_cols`` to
    remap if they don't match the defaults below.
    """
    if val_cols is None:
        val_cols = {"macro_f1": "macro_f1", "macro_recall": "macro_recall", 
                    "macro_precision": "macro_precision", "macro_auprc": "macro_auprc"}
    if oot_1_cols is None:
        oot_1_cols = {"macro_f1": "macro_f1", "macro_recall": "macro_recall",
                      "macro_precision": "macro_precision", "macro_auprc": "macro_auprc"}
    if oot_2_cols is None:
        oot_2_cols = {"macro_f1": "macro_f1", "macro_recall": "macro_recall",
                      "macro_precision": "macro_precision", "macro_auprc": "macro_auprc"}

    metrics = ("macro_f1", "macro_recall", "macro_precision", 'macro_auprc')
    rows = []
    for split, df, cols in (("Validation", val_df, val_cols), ("OOT_2025Q4", oot_1_df, oot_1_cols), 
                            ("OOT_2026Q1", oot_2_df, oot_2_cols)):
        row = {"split": split, "n_runs": len(df)}
        for metric in metrics:
            vals = df[cols[metric]]
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std() if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_val_oot_comparison(
    summary: pd.DataFrame,
    save_path: Optional[Path] = None,
    legend_loc="best",
    title: Optional[str] = None,
    width = 0.25,
) -> None:
    """Grouped bar chart of macro-F1 / macro-Recall / macro-Precision, Validation vs OOT.

    ``summary`` is the output of :func:`summarize_val_oot_comparison` (one row per split,
    with ``{metric}_mean``/``{metric}_std`` columns). Same visual convention as
    :func:`plot_loss_comparison`: grouped bars, ±1-std error whiskers, bold in-bar labels.
    """
    import matplotlib.pyplot as plt

    def _label_bars(container):
        for rect in container:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h * 0.90),
                    ha="center",
                    va="top",
                    fontsize=11,
                    color="black",
                    fontweight="bold",
                    clip_on=True,
                )

    labels = summary["split"].tolist()
    f1_means = summary["macro_f1_mean"].tolist()
    recall_means = summary["macro_recall_mean"].tolist()
    precision_means = summary["macro_precision_mean"].tolist()
    f1_std = summary.get("macro_f1_std", pd.Series(0.0, index=summary.index)).tolist()
    recall_std = summary.get("macro_recall_std", pd.Series(0.0, index=summary.index)).tolist()
    precision_std = summary.get("macro_precision_std", pd.Series(0.0, index=summary.index)).tolist()

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))
    width = width
    err_kw = dict(capsize=4, ecolor="#333333", elinewidth=1.2)

    if "macro_auprc_mean" in summary.columns:
        auprc_means = summary["macro_auprc_mean"].tolist()
        auprc_std = summary.get("macro_auprc_std", pd.Series(0.0, index=summary.index)).tolist()
        bars_auprc = ax.bar(x + 2 * width, auprc_means, width=width, yerr=auprc_std, color="#8172B2",
                        label="Macro-AUPRC", error_kw=err_kw)    
        

        bars_f1 = ax.bar(x - width, f1_means, width=width, yerr=f1_std, color="#4C72B0",
                        label="Macro-F1", error_kw=err_kw)
        bars_r = ax.bar(x, recall_means, width=width, yerr=recall_std, color="#55A868",
                        label="Macro-Recall", error_kw=err_kw)
        bars_p = ax.bar(x + width, precision_means, width=width, yerr=precision_std, color="#C44E52",
                        label="Macro-Precision", error_kw=err_kw)
    
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylabel("Macro score", fontsize=14)
        if title:
            ax.set_title(title, fontsize=15)
        
        tops = [m + s for m, s in zip(
            (*f1_means, *recall_means, *precision_means, *auprc_means),
            (*f1_std, *recall_std, *precision_std, *auprc_std),
        )]
        ymax = max([0.0, *tops])
        ax.set_ylim(0, min(1.0, ymax * 1.2 if ymax > 0 else 1.0))
        ax.legend(loc=legend_loc, ncol=3, fontsize=14, frameon=False, title=None)

        _label_bars(bars_f1)
        _label_bars(bars_r)
        _label_bars(bars_p)
        _label_bars(bars_auprc)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved figure to {save_path}")
        plt.show()
    
    else:
        bars_f1 = ax.bar(x - width, f1_means, width=width, yerr=f1_std, color="#4C72B0",
                        label="Macro-F1", error_kw=err_kw)
        bars_r = ax.bar(x, recall_means, width=width, yerr=recall_std, color="#55A868",
                        label="Macro-Recall", error_kw=err_kw)
        bars_p = ax.bar(x + width, precision_means, width=width, yerr=precision_std, color="#C44E52",
                        label="Macro-Precision", error_kw=err_kw)
    
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylabel("Macro score", fontsize=14)
        if title:
            ax.set_title(title, fontsize=15)
        
        tops = [m + s for m, s in zip(
            (*f1_means, *recall_means, *precision_means),
            (*f1_std, *recall_std, *precision_std),
        )]
        ymax = max([0.0, *tops])
        ax.set_ylim(0, min(1.0, ymax * 1.2 if ymax > 0 else 1.0))
        ax.legend(loc=legend_loc, ncol=3, fontsize=14, frameon=False, title=None)

        _label_bars(bars_f1)
        _label_bars(bars_r)
        _label_bars(bars_p)
       
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved figure to {save_path}")
        plt.show()


def summary_val_oot_experiment(
    val_df: pd.DataFrame,
    oot_1_df: pd.DataFrame,
    oot_2_df: pd.DataFrame,
    val_cols: Optional[Dict[str, str]] = None,
    oot_1cols: Optional[Dict[str, str]] = None,
    oot_2_cols: Optional[Dict[str, str]] = None,
    output_name: str = "val_oot_macro_comparison.png",
    output_path: Optional[Path] = None,
    verbose: bool = True,
    plot: bool = True,
    legend_loc="best",
    title: Optional[str] = None,
    width: float = 0.3
) -> pd.DataFrame:
    """Summarize + plot macro-F1/Recall/Precision, Validation vs OOT, mean ± std over seeds."""
    summary = summarize_val_oot_comparison(val_df, oot_1_df, oot_2_df, 
                            val_cols=val_cols, oot_1_cols=oot_1cols, oot_2_cols=oot_2_cols)

    if verbose:
        print("\n" + "=" * 72)
        print("Validation vs OOT macro-metric summary (mean ± std over seeds)")
        print("=" * 72)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if plot and len(summary):
        try:
            plot_val_oot_comparison(
                summary,
                save_path=(f"{output_path}/{output_name}") if output_path else None,
                legend_loc=legend_loc,
                title=title,
                width=width,
            )
        except Exception as exc:
            print(f"(val/OOT comparison plot skipped: {exc})")

    return summary


# ---------------------------------------------------------------------------
# Robust (de)serialization of run artifacts
# ---------------------------------------------------------------------------

def _artifacts_to_picklable(
    artifacts: Dict[str, Dict], *, strip_models: bool = True
) -> Dict[str, Dict]:
    """
    Convert the nested ``artifacts`` dict into an identity-independent, picklable form.

    Fixes the ``PicklingError: ... it's not the same object as
    loss_experiments.multihot_loss_comparison.RunResult`` that happens when the module
    is reloaded (e.g. under ``%autoreload``) so the stored ``RunResult`` instances point
    at a stale class object:
      - ``RunResult`` -> plain ``dict`` via ``asdict`` (works even on stale instances,
        since ``asdict`` uses the fields of whatever dataclass the instance is).
      - live ``nn.Module`` -> CPU ``state_dict`` under ``"model_state_dict"`` when
        ``strip_models=True`` (portable across sessions; the model class need not be
        importable/identical at load time). Set ``strip_models=False`` to keep the live
        model object (only safe to reload in a session where its class is importable).
    """
    clean: Dict[str, Dict] = {}
    for loss_name, seed_map in artifacts.items():
        clean[loss_name] = {}
        for seed, entry in seed_map.items():
            entry = dict(entry)  # shallow copy so we don't mutate the caller's dict
            res = entry.get("result")
            if is_dataclass(res) and not isinstance(res, type):
                entry["result"] = asdict(res)
            model = entry.get("model")
            if strip_models and isinstance(model, nn.Module):
                entry.pop("model")
                entry["model_state_dict"] = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
            clean[loss_name][seed] = entry
    return clean


def save_loss_artifacts(artifacts: Dict[str, Dict], path, *, strip_models: bool = True) -> Path:
    """
    Robustly persist ``run_loss_comparison_experiment`` artifacts.

    Example
    -------
        save_loss_artifacts(loss_label_artifacts, "outputs/loss_label_artifacts.pt")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_artifacts_to_picklable(artifacts, strip_models=strip_models), path)
    return path


def load_loss_artifacts(path, *, as_runresult: bool = True) -> Dict[str, Dict]:
    """
    Load artifacts saved by :func:`save_loss_artifacts`.

    With ``as_runresult=True`` the plain-dict ``result`` entries are rebuilt into
    ``RunResult`` objects (using the *current* class, so no identity mismatch). Model
    weights come back under ``"model_state_dict"``; rebuild the module with your
    ``model_factory`` and call ``model.load_state_dict(entry["model_state_dict"])``.
    """
    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    if as_runresult:
        for seed_map in obj.values():
            for entry in seed_map.values():
                res = entry.get("result")
                if isinstance(res, dict):
                    try:
                        entry["result"] = _run_result_from_mapping(res)
                    except TypeError:
                        # Saved schema differs from the current RunResult
                        # (fields added/removed) — keep the plain dict rather than crash.
                        pass
    return obj