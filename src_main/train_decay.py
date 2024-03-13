import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *


# Need to adjust the training function to include the learning rate scheduler
def train_model_decay(model, optimizer, lr_scheduler, target, xy_grid, epochs=2000, display_epochs=[0, 500, 1000, 1500, 1900, 2000]):
    seed_everything()
    snapshots = []
    train_psnrs = []
    test_psnrs = []
    image_processor = ImageProcessor(device=xy_grid.device)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        optimizer.zero_grad()
        output = model(xy_grid)
        loss = nn.functional.l1_loss(target, output)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_psnr = calculate_psnr(target.cpu(), output.cpu())
            train_psnrs.append(train_psnr)
            test_psnr = calculate_psnr(target.cpu(), output.cpu())
            test_psnrs.append(test_psnr)

        if epoch in display_epochs:
            generated_image = image_processor.tensor_to_numpy(output[0])
            snapshots.append(generated_image)

        lr_scheduler.step()

    ground_truth_image = image_processor.tensor_to_numpy(target[0])
    snapshots.append(ground_truth_image)

    plt.figure(figsize=(12, 6))

    # Plot final reconstruction
    plt.subplot(1, 2, 1)
    plt.imshow(snapshots[-2])
    plt.title('Final Reconstruction')

    # Plot ground truth
    plt.subplot(1, 2, 2)
    plt.imshow(snapshots[-1])
    plt.title('Ground Truth')

    plt.show()
    # Plot reconstructed images
    plt.figure(figsize=(24, 10))
    for i, image in enumerate(snapshots):
        plt.subplot(1, len(snapshots), i + 1)
        plt.imshow(image)
        plt.title('Epoch {}'.format(display_epochs[i]) if i < len(snapshots) - 1 else 'Ground Truth')
    plt.show()

    # Plot PSNR curves
    plt.figure(figsize=(16, 6))
    epochs_range = range(epochs)

    plt.subplot(121)
    plt.plot(epochs_range, train_psnrs, label='Train PSNR')
    plt.title('Train PSNR over Epochs')
    plt.ylabel('PSNR (dB)')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(122)
    plt.plot(epochs_range, test_psnrs, label='Test PSNR')
    plt.title('Test PSNR over Epochs')
    plt.ylabel('PSNR (dB)')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()