import torch

def evaluate_gan(generator, test_loader, device):
    generator.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for images, voxels in test_loader:
            images = images.to(device)
            voxels = voxels.to(device)
            generated_voxels = generator(images)
            loss = criterion(generated_voxels, voxels)
            total_loss += loss.item()

    generator.train()
    return total_loss / len(test_loader)