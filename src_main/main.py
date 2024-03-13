from parser import args
import torch
import numpy as np
from model import Model
from optimizer import get_adam_optimizer
from utils import ImageProcessor, train_model
from scheduler import build_scheduler
from train_decay import train_model_decay


# High Resolution Lion
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    image_processor = ImageProcessor(device=device)
    target_image_url = args.image_url
    target = image_processor.get_image_from_url(target_image_url, crop_center=args.crop_center)

    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords, coords), axis=-1)
    xy_grid = torch.tensor(xy_grid, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)

    # Experiment without Gaussian transformations
    print("Training without Gaussian transformations...")
    model_without_gaussian = Model(device, use_gaussian_transform=False)
    optimizer_without_gaussian = get_adam_optimizer(model_without_gaussian.parameters(), lr=args.lr, wd=args.weight_decay)
    train_model(model_without_gaussian, optimizer_without_gaussian, target, xy_grid)

    # Experiment with Gaussian transformations
    print("Training with Gaussian transformations...")
    model_with_gaussian = Model(device, use_gaussian_transform=True, input_channels=args.input_channels, output_channels=args.output_channels)
    optimizer_with_gaussian = get_adam_optimizer(model_with_gaussian.parameters(), lr=args.lr, wd=args.weight_decay)
    train_model(model_with_gaussian, optimizer_with_gaussian, target, xy_grid)


# High Resolution Tiger
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    image_processor = ImageProcessor(device=device)
    target_image_url = 'https://s1.1zoom.me/big0/158/Tigers_Amur_tigerSnout_498210.jpg'
    target = image_processor.get_image_from_url(target_image_url, crop_center=800)

    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords, coords), axis=-1)
    xy_grid = torch.tensor(xy_grid, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)

    # Experiment without Gaussian transformations
    print("Training without Gaussian transformations...")
    model_without_gaussian = Model(device, use_gaussian_transform=False)
    optimizer_without_gaussian = get_adam_optimizer(model_without_gaussian.parameters(), lr=args.lr, wd=args.weight_decay)
    train_model(model_without_gaussian, optimizer_without_gaussian, target, xy_grid)

    # Experiment with Gaussian transformations
    print("Training with Gaussian transformations...")
    model_with_gaussian = Model(device, use_gaussian_transform=True, input_channels=args.input_channels, output_channels=args.output_channels)
    optimizer_with_gaussian = get_adam_optimizer(model_with_gaussian.parameters(), lr=args.lr, wd=args.weight_decay)
    train_model(model_with_gaussian, optimizer_with_gaussian, target, xy_grid)


# High Resolution Lion (different than first one)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    image_processor = ImageProcessor(device=device)
    target_image_url = 'https://img.freepik.com/premium-photo/african-animals-portrait-majestic-large-lion-suns-rays_124507-50906.jpg'
    target = image_processor.get_image_from_url(target_image_url, crop_center=412)

    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords, coords), axis=-1)
    xy_grid = torch.tensor(xy_grid, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)

    # Experiment without Gaussian transformations
    print("Training without Gaussian transformations...")
    model_without_gaussian = Model(device, use_gaussian_transform=False)
    optimizer_without_gaussian = get_adam_optimizer(model_without_gaussian.parameters(), lr=args.lr, wd=args.weight_decay)
    lr_scheduler = build_scheduler(args, optimizer_without_gaussian)
    train_model_decay(model_without_gaussian, optimizer_without_gaussian, lr_scheduler, target, xy_grid)

    # Experiment with Gaussian transformations
    print("Training with Gaussian transformations...")
    model_with_gaussian = Model(device, use_gaussian_transform=True, input_channels=args.input_channels, output_channels=args.output_channels)
    optimizer_with_gaussian = get_adam_optimizer(model_with_gaussian.parameters(), lr=args.lr, wd=args.weight_decay)
    lr_scheduler = build_scheduler(args, optimizer_with_gaussian)
    train_model_decay(model_with_gaussian, optimizer_with_gaussian, lr_scheduler, target, xy_grid)