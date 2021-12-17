
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from util import Util


class PatientDataset(Dataset):
    """Declaring dataset"""

    def __init__(self, x_df, y):
        self.X_df = x_df
        self.Y = y

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx):
        if torch.cuda.is_available:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        x = torch.tensor(self.X_df.iloc[idx],
                         device=device, dtype=torch.float)
        y = torch.tensor(self.Y[idx], device=device)
        return x, y
