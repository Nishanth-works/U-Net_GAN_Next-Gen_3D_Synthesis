import os
import torch

from config import *
from data.data_loader import load_data
from models.generator import Generator
from models.discriminator import Discriminator
from training.train import train_gan
from training.evaluate import evaluate_gan
from visualization.visualize import visualize_samples

def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = load_data(DATASET_PATH, IMAGE_SIZE, BATCH_SIZE)

    # Initialize the generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Load pretrained models if available
    if os.path.exists(GENERATOR_PATH):
        generator.load_state_dict(torch.load(GENERATOR_PATH))
    if os.path.exists(DISCRIMINATOR_PATH):
        discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH))

    # Train the GAN
    train_gan(
        generator, discriminator, train_loader,
        device, EPOCHS, LR, BETA1, BETA2,
        GENERATOR_PATH, DISCRIMINATOR_PATH
    )

    # Evaluate the GAN
    evaluation_results = evaluate_gan(generator, test_loader, device)
    with open(EVALUATION_PATH, 'w') as f:
        f.write(str(evaluation_results))

    # Visualize samples
    visualize_samples(generator, test_loader, device)

if __name__ == "__main__":
    main()