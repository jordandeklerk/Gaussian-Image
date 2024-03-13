import torch
import torch.nn as nn
from parser import args


class Gaussian(nn.Module):
    def __init__(self, num_input_channels, mapping_size=args.mapping_size, scale=args.scale):
        super(Gaussian, self).__init__()
        self.B = nn.Parameter(torch.randn(num_input_channels, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        x_transformed = torch.matmul(x.permute(0, 2, 3, 1), self.B.to(x.device))
        x_transformed = 2 * np.pi * x_transformed
        return torch.cat([torch.sin(x_transformed), torch.cos(x_transformed)], dim=-1).permute(0, 3, 1, 2)
    
def conv_block(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )

# Essentially an MLP since we're using 1x1 convolutions
class Model(nn.Module):
    def __init__(self, device, use_gaussian_transform=False, input_channels=args.input_channels, output_channels=args.output_channels):
        super(Model, self).__init__()
        layers = []

        if use_gaussian_transform:
            layers.append(Gaussian(input_channels, 128, 10))
            input_channels = 256

        layers.extend([
            conv_block(input_channels, 256, kernel_size=1),
            conv_block(256, 256, kernel_size=1),
            conv_block(256, 256, kernel_size=1),
            nn.Conv2d(256, output_channels, kernel_size=1),
            nn.Sigmoid(),
        ])

        self.model = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        return self.model(x)