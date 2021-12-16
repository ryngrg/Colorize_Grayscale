import math
import torch
import os
from torch.utils.data import Dataset, DataLoader
from colorize_data import ColorizeData
from basic_model import Net
from torch.nn import MSELoss


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
        dataset = ColorizeData()
        # dataloaders
        train_dataset = dataset[:int(len(dataset) * 0.8)]
        iters_per_epoch = math.ceil(len(train_dataset) / self.batch_size)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = dataset[int(len(dataset) * 0.8):]
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        # Model
        model = self.model
        # Loss function to use
        criterion = MSELoss()
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        # train loop
        for epoch in range(self.epochs):
            for i, (X, Y) in enumerate(train_dataloader):
                print(f'epoch {epoch+1}/{self.epochs}, step {i+1}/{iters_per_epoch}, inputs {X.shape}')
                y_hat = model(X)
                loss = criterion(y_hat, Y)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), r"../models/model")

    def validate(self):
        pass
        # Validation loop begin
        # ------
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.