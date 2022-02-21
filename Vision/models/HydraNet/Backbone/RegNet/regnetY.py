"""
regnetY.py - Pytorch implementation of "Designing Network Design Spaces, Radosavovic et al., (CVPR 2020)"
"""

from typing import List, Union

import torch
import torch.optim as optim

import pytorch_lightning as pl
import pycls.models as models
from torchinfo import summary


class HermesRegNetY(pl.LightningModule):
    def __init__(
        self, 
        type: str="400MF", 
        load_pretrained: bool=True, 
        verbose: bool=False) -> None:
        """
        Construct RegNetY model by loading from pycls' model zoo.

        Args:
        - type: String.
            Type of RegNetY model to be loaded.
        - load_pretrained: Bool.
            Determine whether to download pretrained weights. 
            Set to true by default.
        - verbose: Bool.
            Determine whether to report all progress or not. 
            Set to false by default, keeping initialization silent.
        """
        super().__init__()

        # initialize RegNetY
        # For details, please refer to https://github.com/facebookresearch/pycls
        self.network = models.regnety(type, pretrained=load_pretrained).cuda()

        if verbose:
            print("[!] Successfully loaded RegNetY-" + type)

            # print model summary
            summary(self.network, input_size=(3, 224, 224), device="cuda")

    def forward(
        self, 
        x: torch.Tensor,
        return_stage_features: bool=True,
        ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward propagation.

        Args:
        - x: Tensor of shape (B, C, H, W).
            A batch of images.

        Returns:
        - out: Can be either
            (1) Tensor of shape ()
            (2) List of tensors
        """
        if return_stage_features:
            out = self._get_stage_features(x)
        else:
            out = self.network(x)
        return out

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(
        self, 
        train_batch: torch.Tensor, 
        batch_idx: int
        ) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(
        self, 
        val_batch: torch.Tensor, 
        batch_idx: int
        ) -> torch.Tensor:
        raise NotImplementedError

    def _get_stage_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Retrieve feature maps from every stage (including head) of RegNet.

        ㄹeatures will be used in subsequent FPN layer, 
        thus must retain information for gradient flow.

        Args:
        - x: Tensor of shape (B, C, H, W). 
            A batch of images from where feature maps will be extracted.

        Returns:
        - A list of tensors containing feature maps
             after forward propagation through each stage.
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
        output.append(x)

        return output
