import torch as tr
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def compute_metrics(x_rec, x_true, mask, output_th=0.5):
    """
    Calculates the F1 score, accuracy (sequence level and threshold),
    precision, recall, and perplexity.

    Args:
        x_rec (torch.Tensor): Reconstructed sequence [N, C, L].
        x_true (torch.Tensor): True sequence [N, C, L].
        output_th (float): Umbral para binarizar las predicciones (default=0.5).
        mask (torch.Tensor): Mask to filter valid sequences [N, C, L].

    Returns:
        dict: Calculated metrics.
    """
    # Flatten para procesar
    x_true_flat = x_true.view(-1).cpu().numpy()  # Real
    x_rec_flat = (x_rec > output_th).float().view(-1).cpu().numpy()

    # Real sequence length (binarizado)
    mask_flat = mask.view(-1).cpu().numpy().astype(bool)

    x_true_filtered = x_true_flat[mask_flat]
    x_rec_filtered = x_rec_flat[mask_flat]

    # 1. F1-Score
    f1 = f1_score(x_true_filtered, x_rec_filtered, average="weighted", zero_division=0)

    # 2. Accuracy (Threshold-based)
    accuracy_tresh = accuracy_score(x_true_filtered, x_rec_filtered)

    # 3. Accuracy a nivel de secuencia
    seq_match = (x_true == (x_rec > output_th).float()).all(dim=1).all(dim=1)
    seq_accuracy = seq_match.float().mean().item()

    # Devolver todas las métricas
    return {"F1": f1, "Accuracy": accuracy_tresh, "Accuracy_seq": seq_accuracy}
