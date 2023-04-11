import numpy as np
from PIL import Image
from scipy.ndimage import zoom


def preprocess_image(image_path, image_size):
    # Load the image and resize it to the desired size
    image = Image.open(image_path)
    image = image.resize((image_size, image_size), Image.ANTIALIAS)

    # Convert the image to a numpy array and normalize it to the range [-1, 1]
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1

    # Convert the image to a PyTorch tensor and add a channel dimension
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)

    return image

def preprocess_voxels(voxel_path, voxel_resolution):
    # Load the voxel model and resize it to the desired resolution
    voxels = np.load(voxel_path)
    voxels = resize_voxels(voxels, voxel_resolution)

    # Convert the voxel model to a PyTorch tensor
    voxels = torch.tensor(voxels).float()

    return voxels

def resize_voxels(voxels, new_resolution):
    original_resolution = voxels.shape[0]
    zoom_factor = new_resolution / original_resolution

    # Use scipy.ndimage.zoom for resizing
    resized_voxels = zoom(voxels, (zoom_factor, zoom_factor, zoom_factor), order=1)

    # Threshold the resized voxels to binarize the output
    threshold = 0.5
    resized_voxels = (resized_voxels > threshold).astype(np.float32)

    return resized_voxels