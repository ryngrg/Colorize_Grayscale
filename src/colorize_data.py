from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import pandas as pd
from PIL import Image
import numpy as np
import os

from torchvision.transforms.functional import resize

class ColorizeData(Dataset):
    def __init__(self):
        # Initialize dataset, you may use a second dataset for validation if required
        self.filenames = pd.read_csv("filenames.csv")
        self.data_dir = r"../data/"
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __len__(self) -> int:
        # return Length of dataset
        return self.filenames.shape[0]
    
    def __getitem__(self, index: int) -> Tuple(torch.Tensor, torch.Tensor):
        # Return the input tensor and output tensor for training
        file = self.filenames[index]
        filepath = os.path.join(os.getcwd(), self.data_dir, file)
        img = Image.open(filepath, mode="r")
        input_img = self.input_transform(img)
        output_img = self.target_transform(img)
        return input_img, output_img
