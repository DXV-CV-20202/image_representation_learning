from typing import Callable, Optional
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import json

from skimage import io

class IRDataset(Dataset):
    def __init__(
        self,
        descriptor_path: str,
        transform: Optional[Callable] = None,
        train: bool = True
    ):
        self.descriptor_path = descriptor_path
        self.transform = transform
        self.train = train
        self._load_meta()

    def _load_meta(self):
        with open(self.descriptor_path) as f:
            self.descriptor = json.load(f)
        self.labels = set()
        self.index = dict()
        for i, d in enumerate(self.descriptor):
            self.labels.add(d['class_name'])
            if d['class_name'] not in self.index:
                self.index[d['class_name']] = []
            self.index[d['class_name']].append(i)

    def __len__(self):
        return len(self.descriptor)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        np.random.RandomState(1)

        positive_idx = idx
        while positive_idx == idx:
            positive_idx = np.random.choice(self.index[self.descriptor[idx]['class_name']])
        negative_label = np.random.choice(list(self.labels - {self.descriptor[idx]['class_name']}))
        negative_idx = np.random.choice(self.index[negative_label])
        
        anchor_image = io.imread(self.descriptor[idx]['image_path'])
        positive_image = io.imread(self.descriptor[positive_idx]['image_path'])
        negative_image = io.imread(self.descriptor[negative_idx]['image_path'])

        images = [anchor_image, positive_image, negative_image]

        for i, image in enumerate(images):
            if self.transform:
                images[i] = self.transform(image)

        return tuple(images), []