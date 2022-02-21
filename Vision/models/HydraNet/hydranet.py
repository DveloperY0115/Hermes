"""
hydranet.py - Pytorch implementation of HydraNet from Tesla FSD vision system.
"""

from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from torchinfo import summary

from .Backbone.RegNet import HermesRegNetY
from .Backbone.BiFPN import HermesBiFPN


class HermesHydraNet(pl.LightningModule):
    def __init__(
        self, 
        regnet_feature_dims: List[int], 
        bifpn_feature_dim: int = 64, 
        verbose: bool = False
    ) -> None:
        """
        Construct Hydra network.

        Args:
        - regnet_feature_dims: A list of integers.
            Number of channels of feature maps obtained from each intermediate
            layer of RegNet.
        - bifpn_feature_dim: Integer.
            Number of channels of BiFPN feature maps. Set to 64 by default.
        - verbose: Determine whether to report all progress or not. 
            Set to false by default, keeping initialization silent.
        """
        super().__init__()

        # RegNetY-400MF
        self.regnet = HermesRegNetY()

        # convolution layers for matching the number of channels
        # TODO: Replace hard-coded numbers to variables
        conv_blocks = []
        for feature_dim in regnet_feature_dims:
            conv_blocks.append(
                nn.Conv2d(
                    feature_dim,
                    bifpn_feature_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
        self.conv_blocks = nn.ModuleList(conv_blocks)

        bn_blocks = [nn.BatchNorm2d(bifpn_feature_dim)] * len(conv_blocks)
        self.bn_blocks = nn.ModuleList(bn_blocks)

        # Bi-FPN module with feature dimensionality 64
        self.bifpn = HermesBiFPN(feature_dim=bifpn_feature_dim)

        if verbose:
            print("[!] Successfully initialized HydraNet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
        - x: Tensor of shape (*, C, H, W). 
            A batch of images from which features will be extracted.

        Returns:
        - 
        """
        # RegNet forward-pass
        regnet_features = self.regnet(x, return_stage_features=True)

        # extract stem & head features
        stem_feature = regnet_features.pop(0)
        head_feature = regnet_features.pop()

        bifpn_input = []
        for feature, conv, bn in zip(regnet_features, self.conv_blocks, self.bn_blocks):
            bifpn_input.append(bn(conv(feature)))

        # BiFPN forward-pass
        multi_scale_features = self.bifpn(bifpn_input)

        return multi_scale_features[-1]

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(
        self, 
        train_batch: torch.Tensor, 
        batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(
        self, 
        val_batch: torch.Tensor, 
        batch_idx: int) -> torch.Tensor:
        raise NotImplementedError
