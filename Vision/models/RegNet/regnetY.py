"""
regnetY.py - Pytorch implementation of "Designing Network Design Spaces, Radosavovic et al., (CVPR 2020)"
"""

import torch
import torch.optim as optim

import pycls.models as models
import pytorch_lightning as pl
from torchsummary import summary


class HermesRegNetY(pl.LightningModule):
    def __init__(self, type: str, load_pretrained: bool = True, verbose=False):
        """
        Construct RegNetY model by loading from pycls' model zoo.

        Args:
        - type: Type of RegNetY model to be loaded.
        - load_pretrained: Determine whether to download pretrained weights. Set to true by default.
        - verbose: Determine whether to report all progress or not. Set to false by default, keeping initialization silent.
        """
        super().__init__()

        # initialize RegNetY
        # For details, please refer to https://github.com/facebookresearch/pycls
        self.network = models.regnety(type, pretrained=load_pretrained).cuda()

        if verbose:
            print("[!] Successfully loaded RegNetY-" + type)

            # print model summary
            summary(self.network, input_size=(3, 224, 224), device="cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
        - x: A tensor of shape (*, C, H, W) representing a batch of images. Typically W = H = 224.
        """
        out = self.network(x)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError
