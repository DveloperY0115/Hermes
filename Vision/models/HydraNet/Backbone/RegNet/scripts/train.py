"""
train.py - Training script for HermesRegNet models
"""

import os
import sys
import torch
import torch.utils.data as data
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import pytorch_lightning as pl

sys.path.append(".")
sys.path.append("..")

from Vision.models.HydraNet.Backbone.RegNet.regnetX import HermesRegNetX
from Vision.models.HydraNet.Backbone.RegNet.regnetY import HermesRegNetY


def main():

    # initialize summary writer for model visualization
    log_dir = "Vision/models/HydraNet/Backbone/RegNet/scripts/logs"

    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # initialize model
    regnetX = HermesRegNetX()
    regnetY = HermesRegNetY()

    _ = regnetY(torch.zeros((1, 3, 224, 224)).cuda(), return_stage_features=True)

    # draw computational graph(s)
    writer.add_graph(regnetX, torch.zeros(1, 3, 224, 224).cuda())
    writer.add_graph(regnetY, torch.zeros(1, 3, 224, 224).cuda())
    writer.close()

    # prepare datasets
    # train_dataset, valid_dataset = get_imagenet_dataset()

    # train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # valid_loader = data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # trainer = pl.Trainer(gpus=[0], precision=16, limit_train_batches=0.5)
    # trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
