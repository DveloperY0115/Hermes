"""
hydranet.py - Pytorch implementation of HydraNet from Tesla FSD vision system.
"""

from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from torchinfo import summary

from Vision.models.HydraNet.Backbone.RegNet import HermesRegNetY
from Vision.models.HydraNet.Backbone.BiFPN import HermesBiFPN, bifpn


class HermesHydraNet(pl.LightningModule):
    def __init__(
        self, regnet_feature_dims: List, bifpn_feature_dim: int = 64, verbose: bool = False
    ) -> None:
        """
        Construct Hydra network.

        Args:
        - verbose: Determine whether to report all progress or not. Set to false by default, keeping initialization silent.
        """
        super().__init__()

        # RegNetY-400MF
        self.regnet = HermesRegNetY()

        # convolution layers for matching the number of channels
        # TODO: Replace hard-coded numbers to variables
        conv_blocks = []
        conv_blocks.append(
            nn.Conv2d(
                regnet_feature_dims[0],  # 48
                bifpn_feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        conv_blocks.append(
            nn.Conv2d(
                regnet_feature_dims[1],  # 104
                bifpn_feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        conv_blocks.append(
            nn.Conv2d(
                regnet_feature_dims[2],  # 208
                bifpn_feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        conv_blocks.append(
            nn.Conv2d(
                regnet_feature_dims[3],  # 440
                bifpn_feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        self.conv_blocks = nn.ModuleList(conv_blocks)

        bn_blocks = []
        for _ in range(len(conv_blocks)):
            bn_blocks.append(nn.BatchNorm2d(bifpn_feature_dim))
        self.bn_blocks = nn.ModuleList(bn_blocks)

        # Bi-FPN module with feature dimensionality 64
        self.bifpn = HermesBiFPN(feature_dim=bifpn_feature_dim)

        # if verbose:
        # summary(self, input_size=(3, 224, 224), device="cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
        - x: A tensor of shape (*, C, H, W) representing a batch of images from which features will be extracted.

        Returns:
        -
        """
        # RegNet forward-pass
        regnet_features = self.regnet.get_stage_features(x)

        # extract stem & head features
        stem_feature = regnet_features.pop(0)
        head_feature = regnet_features.pop()

        bifpn_input = []
        for idx, feature in enumerate(regnet_features):
            bifpn_input.append(self.bn_blocks[idx](self.conv_blocks[idx](feature)))

        # BiFPN forward-pass
        multi_scale_features = self.bifpn(bifpn_input)

        return multi_scale_features[-1]

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError
