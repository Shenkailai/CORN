import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from basemodel import BaseModel
from fr import FullReferenceHead 
from nr import NoReferenceHead 

class CORN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.base_model = BaseModel()
        
        self.fr_head = FullReferenceHead()

        self.nr_head = NoReferenceHead()
        
    def forward(self, x_i, r_j=None):
        e_i = self.base_model(x_i) 

        if r_j is not None:
            e_j = self.base_model(r_j) 
            f_ij = self.fr_head(e_i, e_j)
            n_i = self.nr_head(e_i) 
            return f_ij, n_i, e_i, e_j 
        else:
            n_i = self.nr_head(e_i) 
            return n_i, e_i  
