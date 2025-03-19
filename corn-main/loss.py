import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
F.smooth_l1_loss
# Smoothed L1 Loss function as defined in the paper
class SmoothedL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff <= self.beta,
            # 0.5 * diff ** 2 / self.beta,
            diff ** 2 / self.beta,
            # diff - 0.5 * self.beta
            2 * diff - self.beta
        )
        return loss.mean()