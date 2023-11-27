import torch
from torch.utils.data import Dataset
from skimage import io, transform
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = io.imread(self.df.file_path[index])
        label = self.df.label[index]

        pi = Image.fromarray(image)

        if (self.transform):
            image = self.transform(pi)

        image = np.array(image)

        # if self.transform:
        #     image = self.transform(image)
            
        return image, label
