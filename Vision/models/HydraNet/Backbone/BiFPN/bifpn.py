"""
bifpn.py - Pytorch implementation of BiFPN introduced in "EfficientDet: Scalable and Efficient Object Detection, Tan et al., (CVPR 2020)"
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from torchinfo import summary


class HermesBiFPN(pl.LightningModule):
    def __init__(
        self, 
        feature_dim: int, 
        num_bifpn_block: int = 3, 
        verbose: bool = False) -> None:
        """
        Construct BiFPN module.

        Args:
        - verbose: Determine whether to report all progress or not. 
            Set to false by default, keeping initialization silent.
        """
        super().__init__()

        bifpn_blocks = [HermesBiFPNBlock(feature_dim)] * num_bifpn_block
        self.bifpn_blocks = nn.ModuleList(bifpn_blocks)

        if verbose:
            print("[!] Successfully initialized BiFPN")

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward propagation.

        Args:
        - x: List of tensors each of which has shape (*, feature_dim)

        Returns:
        - x: List of tensors each of which has shape (*, feature_dim) after top-down / bottom-up feature fushion
        """
        for bifpn_block in self.bifpn_blocks:
            x = bifpn_block(x)

        return x

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError


class HermesBiFPNBlock(pl.LightningModule):
    def __init__(
        self, 
        feature_dim: int, 
        eps: float = 1e-4):
        """
        BiFPN block. A basic building block of Bi-directional Feature Pyramid Network (BiFPN).
        For architecture details, please refer to 'EfficientDet: Scalable and Efficient Object Detection, Tan et al. (CVPR 2020)'

        NOTE: This module is intended to be used with RegNet backbone, 
        not EfficientNet as in the original paper.

        Args:
        - feature_dim: Dimensionality of the input feature(s)
        - eps: Small value used to avoid numerical instability during 'Fast normalized fusion'. Set to 1e-4 by default.
        """
        super().__init__()

        # initialize epsilon
        self.eps = eps

        # convolution layers for top-down pathway
        self.conv_s1_td = HermesDepthwiseConvBlock(
            feature_dim, feature_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv_s2_td = HermesDepthwiseConvBlock(
            feature_dim, feature_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv_s3_td = HermesDepthwiseConvBlock(
            feature_dim, feature_dim, kernel_size=3, stride=1, padding=1
        )
        # self.conv_s4_td = HermesDepthwiseConvBlock(feature_dim, feature_dim)

        # convolution layers for bottom-up pathway
        self.conv_s2_bu = HermesDepthwiseConvBlock(
            feature_dim, feature_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv_s3_bu = HermesDepthwiseConvBlock(
            feature_dim, feature_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv_s4_bu = HermesDepthwiseConvBlock(
            feature_dim, feature_dim, kernel_size=3, stride=1, padding=1
        )
        # self.conv_head_bu = HermesDepthwiseConvBlock(feature_dim, feature_dim)

        # initialize weights & activations
        self.w_td = nn.Parameter(torch.randn((3, 2), dtype=torch.float32))
        self.w_bu = nn.Parameter(torch.randn((3, 3), dtype=torch.float32))
        self.w_td_relu = nn.ReLU()
        self.w_bu_relu = nn.ReLU()

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward propagation.

        Args:
        - features: List of tensors from different stages of RegNet backbone.

        Returns:
        - multi_scale_features: List of tensors
        """
        s1, s2, s3, s4 = features

        # top-down
        w_td = self.w_td_relu(self.w_td)
        w_td /= torch.sum(w_td, dim=1, keepdim=True).repeat((1, 2)) + self.eps
        w_bu = self.w_bu_relu(self.w_bu)
        w_bu = torch.sum(w_bu, dim=1, keepdim=True).repeat((1, 3)) + self.eps

        s4_td = s4
        s3_td = self.conv_s3_td(w_td[0, 0] * s3 + w_td[0, 1] * F.interpolate(s4_td, scale_factor=2))
        s2_td = self.conv_s2_td(w_td[1, 0] * s2 + w_td[1, 1] * F.interpolate(s3_td, scale_factor=2))
        s1_td = self.conv_s1_td(w_td[2, 0] * s1 + w_td[2, 1] * F.interpolate(s2_td, scale_factor=2))

        # bottom-up
        s1_out = s1_td
        s2_out = self.conv_s2_bu(
            w_bu[0, 0] * s2
            + w_bu[0, 1] * s2_td
            + w_bu[0, 2] * F.interpolate(s1_out, scale_factor=0.5, recompute_scale_factor=False)
        )
        s3_out = self.conv_s3_bu(
            w_bu[1, 0] * s3
            + w_bu[1, 1] * s3_td
            + w_bu[1, 2] * F.interpolate(s2_out, scale_factor=0.5, recompute_scale_factor=False)
        )
        s4_out = self.conv_s4_bu(
            w_bu[2, 0] * s4
            + w_bu[2, 1] * s4_td
            + w_bu[2, 2] * F.interpolate(s3_out, scale_factor=0.5, recompute_scale_factor=False)
        )

        multi_scale_features = [s1_out, s2_out, s3_out, s4_out]

        return multi_scale_features


class HermesDepthwiseConvBlock(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        """
        Depthwise separable convolution layer with batch normalization and ReLU activation.

        Args:
        - in_channels: Number of channels in the input image
        - out_channels: Number of channels produced by the convolution
        - kernel_size: Size of the convolving kernel. Set to 1 by default.
        - stride: Stride of the convolution. Set to 1 by default.
        - padding: Padding added to all four sides of the input. Set to 0 by default.
        - dilation: Spacing between kernel elements. Set to 1 by default.
        """
        super().__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.actvn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
        - x:

        Returns:
        - out
        """
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        out = self.actvn(x)
        return out


class HermesConvBlock(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        """
        Convolution layer with batch normalization and ReLU activation.

        Args:
        - in_channels: Number of channels in the input image
        - out_channels: Number of channels produced by the convolution
        - kernel_size: Size of the convolving kernel. Set to 1 by default.
        - stride: Stride of the convolution. Set to 1 by default.
        - padding: Padding added to all four sides of the input. Set to 0 by default.
        - dilation: Spacing between kernel elements. Set to 1 by default.
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.actvn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
        - x:

        Returns:
        - out:
        """
        x = self.conv(x)
        x = self.bn(x)
        out = self.actvn(x)
        return out
