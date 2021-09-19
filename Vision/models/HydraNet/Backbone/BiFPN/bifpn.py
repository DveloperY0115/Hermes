"""
bifpn.py - Pytorch implementation of BiFPN introduced in "EfficientDet: Scalable and Efficient Object Detection, Tan et al., (CVPR 2020)"
"""

import torch
import torch.optim as optim

import pytorch_lightning as pl
from torchsummary import summary


class HermesBiFPN(pl.LightningModule):
    def __init__(self, verbose: bool = False) -> None:
        """
        Construct BiFPN module.

        Args:
        - verbose: Determine whether to report all progress or not. Set to false by default, keeping initialization silent.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError
