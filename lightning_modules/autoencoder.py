# model.py

import torch
import pytorch_lightning as pl
from dataset_classes.datasets import MyDataset
from models.encoders import Encoder
from typing import Dict

class AutoEncoder(pl.LightningModule):
    """ 
    AutoEncoder model class using PyTorch Lightning.

    Args:
        hparams (Dict): Dictionary of hyperparameters.
    """
    def __init__(self, hparams: Dict):
        # TODO: Define complete model architecture.
        super().__init__()
        self.save_hyperparameters(hparams)
        self.encoder = Encoder(self.hparams.input_size, self.hparams.output_size)

    def forward(self, x):
        # TODO: Define forward pass.
        return self.encoder(x)
    
    def loss(self, y_hat, y):
        # TODO: Define loss function.
        return torch.nn.functional.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError("Invalid optimizer.")
        return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(MyDataset('train'), batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(MyDataset('val'), batch_size=self.hparams.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(MyDataset('test'), batch_size=self.hparams.batch_size, shuffle=False)