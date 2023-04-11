import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def display_2d_image(image):
    image = (image + 1) / 2  # Unnormalize the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_3d_voxels(voxels, threshold=0.5):
    # Convert the voxel model to a binary format using the threshold
    binary_voxels = (voxels > threshold).astype(np.float32)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the voxels where the binary_voxels value is 1
    filled_voxels = np.where(binary_voxels == 1)
    ax.scatter(filled_voxels[0], filled_voxels[1], filled_voxels[2], zdir='z', c='b')

    plt.show()

def visualize_results(generator, test_loader, device):
    # Take one sample from the test_loader
    images, _ = next(iter(test_loader))
    image = images[0].to(device)

    # Generate the 3D voxel model using the trained generator
    with torch.no_grad():
        generated_voxels = generator(image.unsqueeze(0))
        generated_voxels = generated_voxels.squeeze().cpu().numpy()

    # Display the 2D input image and the 3D voxel model
    display_2d_image(image.permute(1, 2, 0).cpu().numpy())
    display_3d_voxels(generated_voxels)