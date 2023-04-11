import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.data_processor import preprocess_image, preprocess_voxels

class Paired2D3DDataset(Dataset):
    def __init__(self, dataset_path, image_size, voxel_resolution):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.voxel_resolution = voxel_resolution
        self.samples = self._load_samples()

    def _load_samples(self):
        # Load the list of samples from the dataset directory
        # Each sample should have a 2D image file and a 3D voxel file
        # (e.g., in numpy format or another suitable format)
        samples = []
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            for sample in os.listdir(category_path):
                sample_path = os.path.join(category_path, sample)
                samples.append(sample_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        image_path = os.path.join(sample_path, "image.png")
        voxel_path = os.path.join(sample_path, "voxels.npy")

        # Load and preprocess the 2D image and 3D voxel model
        image = preprocess_image(image_path, self.image_size)
        voxels = preprocess_voxels(voxel_path, self.voxel_resolution)

        return image, voxels

def load_data(dataset_path, image_size, batch_size):
    dataset = Paired2D3DDataset(dataset_path, image_size, VOXEL_RESOLUTION)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader