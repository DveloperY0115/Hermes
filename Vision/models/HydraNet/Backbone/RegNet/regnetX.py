"""
regnetX.py - Pytorch implementation of "Designing Network Design Spaces, Radosavovic et al., (CVPR 2020)"
"""
from typing import List

import torch
import torch.optim as optim

import pycls.models as models
import pytorch_lightning as pl
from torchinfo import summary


class HermesRegNetX(pl.LightningModule):
    def __init__(self, type: str = "400MF", load_pretrained: bool = True, verbose=False) -> None:
        """
        Construct RegNetX model by loading from pycls' model zoo.

        Args:
        - type: Type of RegNetX model to be loaded.
        - load_pretrained: Determine whether to download pretrained weights. Set to true by default.
        - verbose: Determine whether to report all progress or not. Set to false by default, keeping initialization silent.
        """
        super().__init__()

        # initialize RegNetX
        # For details, please refer to https://github.com/facebookresearch/pycls
        self.network = models.regnetx(type, pretrained=load_pretrained).cuda()

        if verbose:
            print("[!] Successfully loaded RegNetX-" + type)

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

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def get_stage_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Retrieve feature maps from every stage (including head) of RegNet.

        NOTE: These features will then be used in subsequent FPN layer, thus must retain information for gradient flow.

        Args:
        - x: A tensor of shape (*, C, H, W). Used for forward propagation.

        Returns:
        - A list of tensors containing feature maps after each forward-pass through each stage.
        """
        output = []

        x = self.network.stem(x)
        output.append(x.clone())

        x = self.network.s1(x)
        output.append(x.clone())

        x = self.network.s2(x)
        output.append(x.clone())

        x = self.network.s3(x)
        output.append(x.clone())

        x = self.network.s4(x)
        output.append(x.clone())

        x = self.network.head(x)
        output.append(x.clone())

        return output
