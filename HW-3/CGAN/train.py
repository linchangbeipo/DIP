import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PairedDataset
from network import Generator, Discriminator, ResGenerator
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch_L1(G_model, dataloader, G_optimizer, L1_loss, device, epoch, num_epochs):
    G_model.train()
    G_loss = 0.0
    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)
        im1 = image_rgb
        im2 = image_semantic

        # 优化生成器
        G_optimizer.zero_grad()
        G_out = G_model(im1)
        G_L1_loss = L1_loss(G_out, im2)
        G_L1_loss.backward()
        G_optimizer.step()

        with torch.no_grad():
            G_loss += G_L1_loss.item()
            if epoch % 5 == 0 and i == 0:
                save_images(im1, im2, G_out, 'train_results/L1+GANloss+res', epoch)

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Train: L1_Loss: {G_L1_loss.item():.4f}')


def train_one_epoch(G_model, D_model, dataloader, G_optimizer, D_optimizer, L1_loss, CGan_loss, device, epoch, num_epochs, lam):
    """
    Train the model for one epoch.

    Args:
        G_model (nn.Module): The Generator model.
        D_model (nn.Module): The Discriminator model.
        dataloader (DataLoader): DataLoader for the training data.
        G_optimizer (Optimizer): Optimizer for updating G_model parameters.
        D_optimizer (Optimizer): Optimizer for updating D_model parameters.
        L1_loss (Loss): L1_Loss function.
        CGan_loss (Loss): Cross-entropy GAN loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    G_model.train()
    D_model.train()

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)
        im1 = image_rgb
        im2 = image_semantic

        # 优化判定器
        D_optimizer.zero_grad()

        # Real
        real_out = D_model(im2, im1)
        real_loss = CGan_loss(real_out, torch.ones_like(real_out, device=device))
        
        
        # Fake
        G_out = G_model(im1)
        fake_out = D_model(G_out.detach(), im1)
        fake_loss = CGan_loss(fake_out, torch.zeros_like(fake_out, device=device))

        D_total_loss = real_loss + fake_loss
        D_total_loss.backward()
        D_optimizer.step()

        # 优化生成器
        G_optimizer.zero_grad()
        D_G_out = D_model(G_out, im1)
        G_CGAN_loss = CGan_loss(D_G_out, torch.ones_like(D_G_out, device=device))
        G_L1_loss = L1_loss(G_out, im2)

        G_total_loss = G_CGAN_loss + lam * G_L1_loss
        G_total_loss.backward()
        G_optimizer.step()

        with torch.no_grad():
            if epoch % 10 == 0 and i == 0:
                save_images(im1, im2, G_out, 'train_results/L1+GANloss+res', epoch)

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Train: G_L1_Loss: {G_L1_loss.item():.4f}  G_GAN_Loss: {G_CGAN_loss.item():.4f}  D_real_Loss: {real_loss.item():.4f}  D_fake_Loss: {fake_loss.item():.4f}')

def validate(G_model, D_model, dataloader, L1_Loss, CGan_loss, device, epoch, num_epochs, lam):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    G_model.eval()
    D_model.eval()
    G_L1_loss = 0.0
    G_GAN_loss = 0.0
    D_real_loss = 0.0
    D_fake_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)
            im1 = image_rgb
            im2 = image_semantic

            real_out = D_model(im2, im1)
            real_loss = CGan_loss(real_out, torch.ones_like(real_out, device=device))

            G_out = G_model(im1)
            fake_out = D_model(G_out.detach(), im1)
            fake_loss = CGan_loss(fake_out, torch.zeros_like(fake_out, device=device))

            D_fake_loss += fake_loss.item()
            D_real_loss += real_loss.item()


            D_G_out = D_model(G_out, im1)
            CGAN_loss = CGan_loss(D_G_out, torch.ones_like(D_G_out, device=device))
            L1_loss = L1_Loss(G_out, im2)

            G_L1_loss += L1_loss.item()
            G_GAN_loss += CGAN_loss.item()

            if epoch % 10 == 0 and i == 0:
                save_images(im1, im2, G_out, 'val_results/L1+GANloss+res', epoch)


    # Calculate average validation loss
    avg_L1_loss = G_L1_loss / len(dataloader)
    avg_GAN_loss = G_GAN_loss / len(dataloader)
    avg_fake_loss = D_fake_loss / len(dataloader)
    avg_real_loss = D_real_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Valid: G_L1_Loss: {avg_L1_loss:.4f}  G_GAN_Loss: {avg_GAN_loss:.4f}  D_Real_Loss: {avg_real_loss:.4f}  D_Fake_Loss: {avg_fake_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = PairedDataset(list_file='train_list.txt')
    val_dataset = PairedDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    G_model = ResGenerator().to(device)
    D_model = Discriminator().to(device)
    L1_loss = nn.L1Loss()
    CGan_loss = nn.BCEWithLogitsLoss()
    G_optimizer = optim.Adam(G_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    num_epochs = 10000
    for epoch in range(num_epochs):
        lam = 100
        if epoch < -1:
            train_one_epoch_L1(G_model, train_loader, G_optimizer, L1_loss, device, epoch, num_epochs)
        else:
            train_one_epoch(G_model, D_model, train_loader, G_optimizer, D_optimizer, L1_loss, CGan_loss, device, epoch, num_epochs, lam)
        validate(G_model, D_model, val_loader, L1_loss, CGan_loss, device, epoch, num_epochs, lam)

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(G_model.state_dict(), f'checkpoints/L1+GANloss+res/pix2pix_Generatormodel_epoch_{epoch + 1}.pth')
            torch.save(D_model.state_dict(), f'checkpoints/L1+GANloss+res/pix2pix_Discriminatormodel_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
