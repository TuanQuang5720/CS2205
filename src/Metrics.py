import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import device
from typing import Tuple
from EmbeddingSpace import EmbeddingSpace


def k_precision(model: Module,
                sketches_val_loader: DataLoader,
                embedding_space: EmbeddingSpace,
                k: int,
                device: device) -> float:
    """
    Calculate k-precision metric
    Args:
        model: The trained model
        sketches_val_loader: DataLoader for validation sketches
        embedding_space: Precomputed embedding space of photos
        k: Number of top matches to consider
        device: Device to run calculations on
    Returns:
        k-precision accuracy percentage
    """
    correct = 0.0
    samples_val = 0

    model.eval()
    with torch.no_grad():
        for sketches, labels in sketches_val_loader:
            sketches = sketches.to(device)
            labels = labels.to(device)

            # Get top-k matches for each sketch
            _, topk_indices = embedding_space.top_k_batch(sketches, k)

            # Check if any of the top-k matches have the same label
            for i in range(len(topk_indices)):
                # Get the labels of the top-k matches
                match_labels = torch.tensor([
                    embedding_space.get_label(idx.item()) for idx in topk_indices[i]
                ]).to(device)

                # Check if the query label is in the matches
                correct += torch.any(match_labels == labels[i]).float()

            samples_val += len(sketches)

    accuracy = 100. * correct / samples_val
    return accuracy