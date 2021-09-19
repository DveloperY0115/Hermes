"""
resnet.py - Pytorch implementation of "Deep Residual Learning for Image Recognition, He et al., (CVPR 2016)"
"""

import torch
from torch._C import Value
from torch.nn.functional import triplet_margin_loss

import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl
from torchvision.models.resnet import resnet34


class HermesResNet(pl.LightningModule):
    def __init__(self, resnet_depth: int = 34, load_pretrained: bool = True):
        """
        Construct ResNet model by loading from torchvision's ResNet model zoo.

        Args:
        - resnet_depth: A number of hidden layers in the model. Can be one of 18, 34, 50, 101, 152. Set to 34 by default.
        - load_pretrained: Determine whether to download pretrained weights. Set to true by default.
        """
        super().__init__()

        if resnet_depth == 18:
            self.resnet = models.resnet18(pretrained=load_pretrained, progress=True)
        elif resnet_depth == 34:
            self.resnet = models.resnet34(pretrained=load_pretrained, progress=True)
        elif resnet_depth == 50:
            self.resnet = models.resnet50(pretrained=load_pretrained, progress=True)
        elif resnet_depth == 101:
            self.resnet = models.resnet101(pretrained=load_pretrained, progress=True)
        elif resnet_depth == 152:
            self.resnet = models.resnet152(pretrained=load_pretrained, progress=True)
        else:
            raise ValueError("[!] Invalid number of hidden layers.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
        - x: A tensor of shape (*, C, H, W) representing a batch of images. Typically W = H = 224.
        """
        out = self.resnet34(x)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError
