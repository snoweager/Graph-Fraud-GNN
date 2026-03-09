import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_auc_score, precision_recall_curve
from src.models.hetero_gnn import HeteroFraudGNN


# -----------------------------------------------------------------------
# Focal Loss
# -----------------------------------------------------------------------
# Why: Standard BCE + class weights over-corrects on imbalanced data,
# causing the model to flag everything as fraud (99% recall, 3% precision).
#
# Focal Loss down-weights easy negatives (obvious non-fraud) so the model
# focuses on hard, ambiguous cases — exactly where fraud hides.
#
# Formula: FL = -alpha * (1 - p_t)^gamma * log(p_t)
#   gamma=2  → easy examples contribute ~4x less than hard ones
#   alpha    → balances positive/negative class frequency
# -----------------------------------------------------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets):
        targets = targets.float()
        bce     = F.binary_cross_entropy(probs, targets, reduction='none')
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        loss    = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def find_best_threshold(y_true, y_proba):
    """
    Find the probability threshold that maximises F1 score.
    Default 0.5 is almost never optimal on imbalanced datasets.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx  = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1        = f1_scores[best_idx]
    print(f"Best threshold: {best_threshold:.4f}  →  F1: {best_f1:.4f}")
    return best_threshold


def train_gnn(graph, epochs=10):

    print("\nSTEP 6: Training Heterogeneous GNN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph  = graph.to(device)

    # -----------------------------------
    # Model
    # -----------------------------------
    in_channels_dict = {
        node_type: graph.x_dict[node_type].shape[1]
        for node_type in graph.x_dict
    }

    model = HeteroFraudGNN(
        metadata=graph.metadata(),
        hidden_dim=64,
        in_channels_dict=in_channels_dict
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler: reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # -----------------------------------
    # Labels + Focal Loss
    # -----------------------------------
    y          = graph["transaction"].y.to(device)
    criterion  = FocalLoss(alpha=0.25, gamma=2.0)

    num_neg = (y == 0).sum().item()
    num_pos = (y == 1).sum().item()
    print(f"Class distribution — Non-fraud: {num_neg:,}  |  Fraud: {num_pos:,}")
    print(f"Using Focal Loss (alpha=0.25, gamma=2.0)")

    # -----------------------------------
    # Output folder
    # -----------------------------------
    os.makedirs("outputs/training", exist_ok=True)

    losses = []

    # -----------------------------------
    # Training Loop
    # -----------------------------------
    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        probs = model(graph)               # [N] probabilities from Sigmoid
        loss  = criterion(probs, y)

        loss.backward()

        # Gradient clipping — prevents exploding gradients on large graphs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step(loss)

        losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

    # -----------------------------------
    # Save Training Loss Graph
    # -----------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(losses, marker='o')
    plt.title("GNN Training Loss (Focal Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/training/gnn_training_loss.png")
    print("Training loss graph saved.")

    # -----------------------------------
    # Evaluation with threshold tuning
    # -----------------------------------
    model.eval()
    with torch.no_grad():
        probs = model(graph)

    y_true  = y.cpu().numpy()
    y_proba = probs.cpu().numpy()

    # Find best threshold on full data (in production: use val set)
    print("\nTuning decision threshold...")
    best_threshold = find_best_threshold(y_true, y_proba)

    y_pred = (y_proba >= best_threshold).astype(int)
    roc    = roc_auc_score(y_true, y_proba)

    print(f"\nGNN ROC-AUC : {roc:.4f}")
    print(f"Threshold   : {best_threshold:.4f}")

    # Return model artifacts so main.py can pass them to the simulation
    model_kwargs = {
        "hidden_dim":       64,
        "in_channels_dict": in_channels_dict,
        "state_dict":       model.state_dict(),
    }

    model_kwargs["model"] = model   # add this line before return
    
    return y_pred, y_proba, model_kwargs
