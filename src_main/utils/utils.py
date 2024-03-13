import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from parser import args


class ImageProcessor:
    def __init__(self, device):
        self.device = device

    def tensor_to_numpy(self, tensor):
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)

    def get_image_from_url(self, url, crop_center, crop_size=(args.crop_size, args.crop_size)):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        transform = transforms.Compose([
            transforms.CenterCrop(crop_center),
            transforms.Resize(crop_size),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        return img_tensor

def calculate_psnr(target, output):
    mse = ((target - output) ** 2).mean().item()
    return 20 * np.log10(1 / np.sqrt(mse)) if mse != 0 else float('inf')

def seed_everything(seed=args.seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False