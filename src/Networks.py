import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import normalize


class SiameseNetwork(nn.Module):
    def __init__(self, output=16, backbone=models.resnet34(weights=models.ResNet34_Weights.DEFAULT)):
        super(SiameseNetwork, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final FC layer

        # Add custom head for embedding
        in_features = backbone.fc.in_features
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output)
        )

    def forward_once(self, x):
        # x shape should be [batch_size, channels, height, width]

        # Debugging: Print input shape
        print(f"Input shape in forward_once: {x.shape}")

        # Ensure we have proper 4D input [batch, channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Verify channel count
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]} channels")

        return self.head(self.backbone(x).flatten(1))

    def forward(self, x):
        # x shape: [batch_size, num_images, channels, height, width]
        # Process each image in the pair/triplet separately
        original_shape = x.shape
        x = x.reshape(-1, *original_shape[2:])  # Flatten first two dimensions

        features = self.backbone(x)
        features = features.reshape(features.size(0), -1)  # Flatten
        embeddings = self.head(features)

        # Reshape back to [batch_size, num_images, embedding_size]
        embeddings = embeddings.reshape(original_shape[0], original_shape[1], -1)

        return embeddings