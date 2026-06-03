"""
SAR-appropriate augmentations for training only.
All transforms operate on numpy arrays: image (2,H,W) float32, label (H,W) float32.
"""

import numpy as np


class RandomFlip:
    def __call__(self, image: np.ndarray, label: np.ndarray):
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()
        return image, label


class RandomRot90:
    def __call__(self, image: np.ndarray, label: np.ndarray):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        label = np.rot90(label, k=k).copy()
        return image, label


class SpeckleNoise:
    """Multiplicative Gamma noise, shape parameter controls severity."""
    def __init__(self, shape: float = 50.0, p: float = 0.3):
        self.shape = shape
        self.p = p

    def __call__(self, image: np.ndarray, label: np.ndarray):
        if np.random.rand() < self.p:
            noise = np.random.gamma(self.shape, 1.0 / self.shape, size=image.shape).astype(np.float32)
            image = image * noise
        return image, label


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


def train_transform() -> Compose:
    return Compose([
        RandomFlip(),
        RandomRot90(),
        SpeckleNoise(shape=50.0, p=0.3),
    ])
