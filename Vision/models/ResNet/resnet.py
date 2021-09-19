"""
resnet.py - Pytorch implementation of "Deep Residual Learning for Image Recognition, He et al., (CVPR 2016)"
"""

import torch
import torch.optim as optim

import pycls.models as models
import pytorch_lightning as pl


class HermesResNet(pl.LightningModule):
    def __init__(self, resnet_depth: int = 50, load_pretrained: bool = True):
        """
        Construct ResNet model by loading from torchvision's ResNet model zoo.

        Args:
        - resnet_depth: A number of hidden layers in the model. Can be one of 50, 101, 152. Set to 50 by default.
        - load_pretrained: Determine whether to download pretrained weights. Set to true by default.
        """
        super().__init__()

        if resnet_depth == 50:
            self.resnet = models.resnet("50", pretrained=load_pretrained)
        elif resnet_depth == 101:
            self.resnet = models.resnet("101", pretrained=load_pretrained)
        elif resnet_depth == 152:
            self.resnet = models.resnet("152", pretrained=load_pretrained)
        else:
            raise ValueError("[!] Invalid number of hidden layers.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
        - x: A tensor of shape (*, C, H, W) representing a batch of images. Typically W = H = 224.
        """
        out = self.resnet(x)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError
