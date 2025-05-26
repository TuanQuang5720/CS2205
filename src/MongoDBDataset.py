import torch
import random
import base64
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import io
import numpy as np


class MongoDBContrastiveDataset(Dataset):
    def __init__(self, photos_collection, sketches_collection, transform=None, is_train=True):
        self.photos_collection = photos_collection
        self.sketches_collection = sketches_collection
        self.transform = transform
        self.is_train = is_train

        # Get all photos and split into train/val
        all_photos = list(photos_collection.find())
        random.shuffle(all_photos)
        split_idx = int(0.8 * len(all_photos))
        self.photos = all_photos[:split_idx] if is_train else all_photos[split_idx:]

        # Create lookup dictionaries
        self.photo_by_id = {p['common_identifier']: p for p in all_photos}
        self.sketches_by_photo = {}

        for sketch in sketches_collection.find():
            photo_id = sketch['photo']
            if photo_id not in self.sketches_by_photo:
                self.sketches_by_photo[photo_id] = []
            self.sketches_by_photo[photo_id].append(sketch)

    def __len__(self):
        return len(self.photos)

    def _load_image(self, image_data):
        """Helper method to load image from MongoDB binary data"""
        if isinstance(image_data, dict) and '$binary' in image_data:
            image_bytes = base64.b64decode(image_data['$binary']['base64'])
        else:
            image_bytes = image_data

        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if not already (handles grayscale/alpha channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def __getitem__(self, idx):
        photo = self.photos[idx]
        try:
            photo_img = self._load_image(photo['image'])
            photo_class = photo['class']

            # Get sketch
            if random.random() > 0.5:  # 50% same class
                sketch_id = random.choice(photo['sketches'])
                sketch = self.sketches_collection.find_one({'common_identifier': sketch_id})
            else:  # Different class
                while True:
                    random_photo = random.choice(self.photos)
                    if random_photo['class'] != photo_class:
                        sketch_id = random.choice(random_photo['sketches'])
                        sketch = self.sketches_collection.find_one({'common_identifier': sketch_id})
                        break

            sketch_img = self._load_image(sketch['image'])

            if self.transform:
                photo_img = self.transform(photo_img)
                sketch_img = self.transform(sketch_img)

            # Ensure we have 3-channel tensors
            if photo_img.shape[0] != 3:
                photo_img = photo_img[:3]  # Take first 3 channels if more exist
            if sketch_img.shape[0] != 3:
                sketch_img = sketch_img[:3]

            inputs = torch.stack((photo_img, sketch_img))  # [2, 3, H, W]
            target = torch.tensor([int(photo_class != sketch['class'])]).long()

            return inputs, target
        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            return self.__getitem__(random.randint(0, len(self) - 1))


class MongoDBAugmentedContrastiveDataset(MongoDBContrastiveDataset):
    def __init__(self, photos_collection, sketches_collection, transform=None, is_train=True):
        super().__init__(photos_collection, sketches_collection, transform, is_train)

    def __getitem__(self, idx):
        inputs, target = super().__getitem__(idx)
        return torch.stack((self.transform(inputs[0]), self.transform(inputs[1]))), target


class MongoDBTripletDataset(Dataset):
    def __init__(self, photos_collection, sketches_collection, transform=None, is_train=True):
        self.photos_collection = photos_collection
        self.sketches_collection = sketches_collection
        self.transform = transform
        self.is_train = is_train

        # Get all photos and split into train/val
        all_photos = list(photos_collection.find())
        random.shuffle(all_photos)
        split_idx = int(0.8 * len(all_photos))
        self.photos = all_photos[:split_idx] if is_train else all_photos[split_idx:]

        # Create lookup dictionaries
        self.photo_by_id = {p['common_identifier']: p for p in all_photos}
        self.sketches_by_photo = {}

        for sketch in sketches_collection.find():
            photo_id = sketch['photo']
            if photo_id not in self.sketches_by_photo:
                self.sketches_by_photo[photo_id] = []
            self.sketches_by_photo[photo_id].append(sketch)

    def _load_image(self, image_data):
        """Helper method to load image from MongoDB binary data"""
        if isinstance(image_data, dict) and '$binary' in image_data:
            # Handle GridFS-style binary data
            image_bytes = base64.b64decode(image_data['$binary']['base64'])
        else:
            # Handle direct binary data
            image_bytes = image_data
        return Image.open(io.BytesIO(image_bytes))

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, idx):
        photo = self.photos[idx]
        try:
            photo_img = self._load_image(photo['image'])
            photo_class = photo['class']

            # Get positive sketch (same class)
            sketch_id = random.choice(photo['sketches'])
            positive_sketch = self.sketches_collection.find_one({'common_identifier': sketch_id})
            positive_img = self._load_image(positive_sketch['image'])

            # Get negative sketch (different class)
            while True:
                random_photo = random.choice(self.photos)
                if random_photo['class'] != photo_class:
                    sketch_id = random.choice(random_photo['sketches'])
                    negative_sketch = self.sketches_collection.find_one({'common_identifier': sketch_id})
                    break

            negative_img = self._load_image(negative_sketch['image'])

            if self.transform:
                photo_img = self.transform(photo_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            inputs = torch.stack((photo_img, positive_img, negative_img))
            target = torch.tensor([0, 1])  # Anchor-Positive, Anchor-Negative

            return inputs, target
        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            # Return a random valid item if this one fails
            return self.__getitem__(random.randint(0, len(self) - 1))


class MongoDBAugmentedTripletDataset(MongoDBTripletDataset):
    def __init__(self, photos_collection, sketches_collection, transform=None, is_train=True):
        super().__init__(photos_collection, sketches_collection, transform, is_train)

    def __getitem__(self, idx):
        inputs, target = super().__getitem__(idx)
        return torch.stack((self.transform(inputs[0]), self.transform(inputs[1]), self.transform(inputs[2]))), target