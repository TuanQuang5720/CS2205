import torch
from torch.nn import Module
from torch.utils.data import DataLoader


class EmbeddingSpace:
    def __init__(self, model, dataloader, device):
        self.device = device
        self.model = model
        self.model.eval()

        self.embeddings = []
        self.labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Handle both single images and pairs/triplets
                if len(images.shape) == 5:  # [batch, num_images, C, H, W]
                    # Use only the first image of each pair/triplet
                    images = images[:, 0, :, :, :]

                embeddings = model.forward_once(images)
                self.embeddings.append(embeddings)
                self.labels.append(labels)

        self.embeddings = torch.cat(self.embeddings, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def top_k_batch(self, query_images, k):
        """
        Find top-k matches for a batch of query images
        Returns:
            distances: torch.Tensor of distances
            indices: torch.Tensor of indices of top-k matches
        """
        query_embeddings = self.model.forward_once(query_images)

        # Compute pairwise distances between query and database embeddings
        distances = torch.cdist(query_embeddings, self.embeddings)

        # Get top-k smallest distances and their indices
        topk_distances, topk_indices = torch.topk(distances, k, largest=False, dim=1)

        return topk_distances, topk_indices

    def get_label(self, index):
        """Get the label for a given index in the embedding space"""
        return self.labels[index]

    def top_k(self, sketches: torch.Tensor, k: int):
        with torch.no_grad():
            sketch_embeddings = torch.squeeze(self.model.forward_once(sketches.to(self.device)))[None, :]

        distances = torch.cdist(self.embeddings, sketch_embeddings)
        topk_distances, topk_indices = torch.topk(distances, k, largest=False, dim=0)

        return topk_distances, topk_indices

    # def top_k_batch(self, sketches: torch.Tensor, k: int):
    #     with torch.no_grad():
    #         sketch_embeddings = torch.squeeze(self.model.forward_once(sketches.to(self.device)))
    #
    #     distances = torch.cdist(self.embeddings, sketch_embeddings)
    #     topk_distances, topk_indices = torch.topk(distances, k, largest=False, dim=0)
    #
    #     return torch.permute(topk_distances, (1, 0)), torch.permute(topk_indices, (1, 0))
