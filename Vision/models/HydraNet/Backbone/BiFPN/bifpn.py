"""
bifpn.py - Pytorch implementation of BiFPN introduced in "EfficientDet: Scalable and Efficient Object Detection, Tan et al., (CVPR 2020)"
"""

from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from torchsummary import summary


class HermesBiFPN(pl.LightningModule):
    def __init__(self, num_layers=3, verbose: bool = False) -> None:
        """
        Construct BiFPN module.

        Args:
        - verbose: Determine whether to report all progress or not. Set to false by default, keeping initialization silent.
        """
        super().__init__()

        if verbose:
            print("[!] Successfully initialized BiFPN")

            # print model summary
            # summary(sel, input_size=(3, 224, 224), device="cuda")

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward propagation.


        """
        pass

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError


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
            kernel_size,
            stride,
            padding,
            dilation,
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

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

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
