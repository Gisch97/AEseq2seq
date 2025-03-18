import torch as tr
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_metrics(x_rec, x_true, output_th=0.5):
    """
    Calcula las métricas F1, accuracy (nivel de secuencia y threshold),
    precision, recall y perplexity.

    Args:
        x_rec (torch.Tensor): Secuencia reconstruida [N, C, L].
        x_true (torch.Tensor): Secuencia real [N, C, L].
        threshold (float): Umbral para binarizar las predicciones (default=0.5).

    Returns:
        dict: Métricas calculadas.
    """
    # Flatten para procesar
    x_true_flat = x_true.view(-1).cpu().numpy()  # Real
    x_rec_flat = (x_rec > output_th).float().view(-1).cpu().numpy()  # Predicho (binarizado)

    # 1. F1-Score
    f1 = f1_score(x_true_flat, x_rec_flat, average="weighted", zero_division=0)

    # 2. Accuracy (Threshold-based)
    accuracy_tresh = accuracy_score(x_true_flat, x_rec_flat)

    # 3. Accuracy a nivel de secuencia
    seq_match = (x_true == (x_rec > output_th).float()).all(dim=1).all(dim=1)
    seq_accuracy = seq_match.float().mean().item() 

    # Devolver todas las métricas
    return {
        "F1": f1,
        "Accuracy": accuracy_tresh,
        "Accuracy_seq": seq_accuracy
    }