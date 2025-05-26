'''
SCRIPT DESCRIPTION:
This script is responsible for training a neural network that performs base information retrieval using sketches.
Now using MongoDB as data source.
'''

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from Utils import *
from Networks import *
from Losses import *
from EarlyStopper import *
from TrainingFunctions import training_loop
from MongoDBDataset import *
from mongo_connection import db

PHOTOS_COLLECTION = db['Photos']
SKETCHES_COLLECTION = db['Sketches']

# ====================================
#               CONFIG
# ====================================

# Pick a Dataset Type
# For training: MongoDBContrastiveDataset, MongoDBTripletDataset,
#               MongoDBAugmentedContrastiveDataset, MongoDBAugmentedTripletDataset
# For validation: MongoDBContrastiveDataset, MongoDBTripletDataset
TRAIN_DATASET_TYPE = MongoDBContrastiveDataset
TRANSFORMATION = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
VAL_DATASET_TYPE = MongoDBContrastiveDataset

# Pick a Criterion
CRITERION = ContrastiveLoss

# Pick an embedding size
OUTPUT_EMBEDDING = 16

# Weight path
WEIGHT_PATH = f"../weights/full-{OUTPUT_EMBEDDING}-contrastive-resnet34.pth"

# Pick a Margin
MARGIN = 2.0

# Pick an Accuracy Margin
ACCURACY_MARGIN = 0.5

# Pick a K (for the K-Precision)
K = 12

# Pick a Batch Size
BATCH_SIZE = 16

# Pick a Backbone
backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

# Pick a Learning Rate
lr = 1e-4

# Pick a number of Epochs
num_epochs = 500

# ====================================
#                CODE
# ====================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"You're using: {DEVICE}")
torch.set_default_dtype(torch.float32)
fix_random(42)
workers = 0

# Create datasets
si_train_dataset = TRAIN_DATASET_TYPE(
    PHOTOS_COLLECTION,
    SKETCHES_COLLECTION,
    transform=TRANSFORMATION,
    is_train=True
)

si_val_dataset = VAL_DATASET_TYPE(
    PHOTOS_COLLECTION,
    SKETCHES_COLLECTION,
    transform=transforms.ToTensor(),
    is_train=False
)

# For k-precision evaluation, we'll use a subset of sketches
all_sketches = list(SKETCHES_COLLECTION.find())
random.shuffle(all_sketches)
k_acc_sketches = all_sketches[:int(0.05 * len(all_sketches))]

# Create data loaders
si_train_loader = DataLoader(
    si_train_dataset,
    shuffle=True,
    num_workers=workers,
    pin_memory=True,
    batch_size=BATCH_SIZE
)

si_val_loader = DataLoader(
    si_val_dataset,
    shuffle=False,
    num_workers=workers,
    pin_memory=True,
    batch_size=BATCH_SIZE
)

# For k-precision we need all photos and a subset of sketches
images_dataloader = DataLoader(
    MongoDBContrastiveDataset(
        PHOTOS_COLLECTION,
        SKETCHES_COLLECTION,
        transform=transforms.ToTensor(),
        is_train=True  # Use all photos for embedding space
    ),
    shuffle=False,
    num_workers=workers,
    pin_memory=True,
    batch_size=BATCH_SIZE
)

small_sketches_loader = DataLoader(
    MongoDBContrastiveDataset(
        PHOTOS_COLLECTION,
        SKETCHES_COLLECTION,
        transform=transforms.ToTensor(),
        is_train=True  # But we'll manually pass the subset
    ),
    shuffle=False,
    num_workers=workers,
    pin_memory=True,
    batch_size=BATCH_SIZE
)

criterion = CRITERION(MARGIN)
net = SiameseNetwork(output=OUTPUT_EMBEDDING, backbone=backbone).to(DEVICE)

optimizer = Adam(net.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=5, min_delta=0)

history = training_loop(
    num_epochs, optimizer, net,
    si_train_loader, si_val_loader, small_sketches_loader,
    images_dataloader, criterion, K, DEVICE, ACCURACY_MARGIN,
    early_stopping=early_stopper
)

torch.save(net.state_dict(), WEIGHT_PATH)