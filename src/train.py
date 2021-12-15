import math
import torch
import os
from torch.utils.data import Dataset, DataLoader
from colorize_data import ColorizeData
from basic_model import Net


class Trainer:
    def __init__(self):
        self.lr = 0.001
        self.model = Net()
        self.weight_decay = 0
        self.betas = (0.9, 0.999)
        self.batch_size = 4
        self.epochs = 32
        # Define hparams here or load them from a config file

    def train(self):
        # dataloaders
        train_dataset = ColorizeData()
        iters_per_epoch = math.ceil(len(train_dataset) / self.batch_size)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = 
        val_dataloader = 
        # Model
        model = self.model
        # Loss function to use
        criterion = 
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        # train loop
        for epoch in range(self.epochs):
            for i, (X, Y) in enumerate(train_dataloader):
                print(f'epoch {epoch+1}/{self.epochs}, step {i+1}/{iters_per_epoch}, inputs {X.shape}')


    def validate(self):
        pass
        # Validation loop begin
        # ------
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.