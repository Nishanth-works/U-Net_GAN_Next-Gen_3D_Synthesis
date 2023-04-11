import torch
import torch.optim as optim
from torch.autograd import Variable
from training.utils import save_checkpoint

def train_gan(generator, discriminator, train_loader, device, epochs, lr, beta1, beta2, generator_path, discriminator_path):
    # Loss function
    criterion = torch.nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    for epoch in range(epochs):
        for i, (images, voxels) in enumerate(train_loader):
            batch_size = images.size(0)
            real_labels = Variable(torch.ones(batch_size)).to(device)
            fake_labels = Variable(torch.zeros(batch_size)).to(device)

            # Train the generator
            optimizer_G.zero_grad()

            images = Variable(images).to(device)
            generated_voxels = generator(images)
            outputs = discriminator(generated_voxels)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # Train the discriminator
            optimizer_D.zero_grad()

            # Real voxels
            voxels = Variable(voxels).to(device)
            outputs = discriminator(voxels)
            d_real_loss = criterion(outputs, real_labels)

            # Fake voxels
            outputs = discriminator(generated_voxels.detach())
            d_fake_loss = criterion(outputs, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")

        # Save checkpoints
        save_checkpoint(generator, generator_path)
        save_checkpoint(discriminator, discriminator_path)