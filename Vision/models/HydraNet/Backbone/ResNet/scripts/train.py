"""
train.py - Training script for HermesResNet model
"""

import os
import sys
import torch.utils.data as data
from torch.utils.data import dataset
import torchvision.datasets as datasets
import pytorch_lightning as pl

sys.path.append(".")
sys.path.append("..")

from Vision.models.HydraNet.Backbone.ResNet.resnet import HermesResNet


def main():

    # initialize model
    model = HermesResNet("50", verbose=True)

    # prepare datasets
    # train_dataset, valid_dataset = get_imagenet_dataset()

    # train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # valid_loader = data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # trainer = pl.Trainer(gpus=[0], precision=16, limit_train_batches=0.5)
    # trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
