import torch
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import Dataset

class MultiPSFDataset(Dataset):
    def __init__(self, data, labels, indices, class_names=None, split=0.85, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.from_numpy(data).to(self.device)
        self.labels = torch.from_numpy(labels).long().to(self.device)
        self.indices = indices
        self.class_names = class_names or [str(i) for i in range(labels.max() + 1)]
        self.split = split
        self.mode = "train"

        self.train_indices = indices[:int(split * len(indices))]
        self.test_indices = indices[int(split * len(indices)):]

    def __len__(self):
        if self.mode == "train":
            return len(self.train_indices)
        else:
            return len(self.test_indices)
    
    def __getitem__(self, idx):
        if self.mode == "train":
            idx = self.train_indices[idx]
        else:
            idx = self.test_indices[idx]
        return self.data[idx].unsqueeze(0).float(), self.labels[idx]
    
    def save(self, path):
        torch.save({
            'data': self.data.cpu().numpy(),
            'labels': self.labels.cpu().numpy(),
            'indices': self.indices,
            'split': self.split,
            'class_names': self.class_names,
        }, path)
    
    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location='cpu')  # load to CPU
        return cls(
            data=checkpoint['data'],
            labels=checkpoint['labels'],
            indices=checkpoint['indices'],
            split=checkpoint['split'],
            class_names=checkpoint['class_names'],
            device=device,  # move to the correct device when creating the dataset
        )
    
    @classmethod
    def create_from_PSFs(cls, categories, class_names=None, device=None, normalize=False):
        # Get the minimum class size
        min_class_size = min(len(category) for category in categories)
        random_seed = 1 # random seed for reproducibility
        np.random.seed(random_seed)
        # Randomly sample instances from each class to equalize class sizes
        equalized_categories = [category[np.random.randint(category.shape[0], size=min_class_size), ...] 
                            for category in categories]
        # Assign data to classes
        if normalize:
            normalized = np.array([cls.normalize(image) for image in np.concatenate(equalized_categories)])
            data = normalized.astype(np.float16)
        else:
            data = np.concatenate(equalized_categories).astype(np.int32)
        labels = np.concatenate([np.full(min_class_size, i) for i, _ in enumerate(categories)])
        indices = np.random.choice(np.arange(len(data)), size=len(data), replace=False)
        return cls(data, labels, indices, class_names=class_names, device=device)
    
    @classmethod
    def normalize(cls, image):
        """Normalize a numpy array representing an image to 0-1 range"""
        image_min = np.min(image)
        image_max = np.max(image)
        image_range = image_max - image_min
        normalized = (image - image_min) / image_range
        return normalized
    
    def train(self):
        self.mode = "train"
    
    def test(self):
        self.mode = "test"


class PSFDataset(Dataset):
    def __init__(self, data, classes, indices, train, split=0.85):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.classes = classes
        self.indices = indices
        self.train = train
        self.split = split

        if train:
            self.images = torch.from_numpy(data[indices[:int(split*len(indices))]]).unsqueeze(1).to(self.device)
            self.labels = torch.from_numpy(classes[indices[:int(split*len(indices))]]).long().to(self.device)
        else:
            self.images = torch.from_numpy(data[indices[int(split*len(indices)):]]).unsqueeze(1).to(self.device)
            self.labels = torch.from_numpy(classes[indices[int(split*len(indices)):]]).long().to(self.device)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx].float(), self.labels[idx]
    
    def save(self, path):
        torch.save({
            'data': self.data,
            'classes': self.classes,
            'indices': self.indices,
            'train': self.train,
            'split': self.split,
        }, path)
    
    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        return cls(
            data=checkpoint['data'],
            classes=checkpoint['classes'],
            indices=checkpoint['indices'],
            train=checkpoint['train'],
            split=checkpoint['split'],
        )
