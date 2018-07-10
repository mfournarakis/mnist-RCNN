import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()

        self.device = device

        self.encoder=nn.Sequential(
            #1st Conv Layer
            nn.Conv2d(1,24,3),
            nn.BatchNorm2d(24),
            nn.RReLU(),
            #2nd Conv Layer
            nn.Conv2d(24,24,3),
            nn.BatchNorm2d(24),
            nn.RReLU(),
            #3rd Conv Layer
            nn.Conv2d(24,48,3,2),
            nn.BatchNorm2d(48),
            nn.RReLU(),
            #4th Conv Layer
            nn.Conv2d(48,48,3),
            nn.BatchNorm2d(48),
            nn.RReLU(),
            #5th Conv Layer
            nn.Conv2d(48,96,3),
            nn.BatchNorm2d(96),
            nn.RReLU(),
            #6th Conv Layer
            nn.Conv2d(96,96,3),
            nn.BatchNorm2d(96),
            nn.RReLU(),
            #7th Conv Layer
            nn.Conv2d(96,192,3),
            nn.BatchNorm2d(192),
            nn.RReLU(),
            #8th Conv Layer
            nn.Conv2d(192,192,3),
            nn.BatchNorm2d(192),
            nn.RReLU(),
            #9th Conv Lyaer
            nn.Conv2d(192,2,1),
            nn.Sigmoid()
            )
    def forward(self,x):
         return self.encoder(x)



def feature_transformer(input, params,device):
    """For now we assume the params are just a single rotation angle

    Args:
        input: [N,c] tensor, where c = 2*int
        params: [N,1] tensor, with values in [0,2*pi)
    Returns:
        [N,c] tensor
    """
    # First reshape activations into [N,c/2,2,1] matrices
    x = input.view(input.size(0),input.size(1)/2,2,1)
    # Construct the transformation matrix
    sin = torch.sin(params)
    cos = torch.cos(params)
    transform = torch.cat([cos, -sin, sin, cos], 1)
    transform = transform.view(transform.size(0),1,2,2).to(device)
    # Multiply: broadcasting taken care of automatically
    # [N,1,2,2] @ [N,channels/2,2,1]
    output = torch.matmul(transform, x)
    # Reshape and return
    return output.view(input.size())

