"""
regnetX.py - Pytorch implementation of "Designing Network Design Spaces, Radosavovic et al., (CVPR 2020)"
"""

import torch
import torch.optim as optim

import pycls.models as models
import pytorch_lightning as pl
from torchsummary import summary


class HermesRegNetX(pl.LightningModule):
    def __init__(self, type: str, load_pretrained: bool = True, verbose=False):
        """
        Construct RegNetX model by loading from torchvision's ResNet model zoo.

        Args:
        - type: Type of RegNetX model to be loaded.
        - load_pretrained: Determine whether to download pretrained weights. Set to true by default.
        """
        super().__init__()

        self.network = models.regnetx(type, pretrained=load_pretrained).cuda()

        if verbose:
            print("[!] Successfully loaded RegnetX-" + type)

            # print model summary
            summary(self.network, input_size=(3, 224, 224))

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