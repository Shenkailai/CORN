import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader

class NoReferenceHead(nn.Module):
    def __init__(self, embedding_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, e_i):
        out = self.fc1(e_i)
        out = self.relu(out)
        out = self.fc2(out)       
        return out.squeeze() 