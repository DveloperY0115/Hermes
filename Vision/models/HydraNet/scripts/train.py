"""
train.py - Training script for HermesHydraNet
"""

import os
import sys
import torch
import torch.utils.data as data
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import pytorch_lightning as pl
from torchinfo import summary
import fiftyone as fo
import fiftyone.zoo as foz

sys.path.append(".")
sys.path.append("..")

from Vision.models.HydraNet.hydranet import HermesHydraNet


def main():

    # initialize summary writer for model visualization
    log_dir = "Vision/models/HydraNet/scripts/logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # initialize model
    hydranet = HermesHydraNet([48, 104, 208, 440], verbose=True).cuda()
    summary(hydranet, input_size=(256, 3, 224, 224), device="cuda")

    # draw computational graph(s)
    writer.add_graph(hydranet, torch.zeros(256, 3, 224, 224).cuda())
    writer.close()

    # prepare datasets
    # TODO: Design a function for fetching & building datasets
    # dataset_detection = foz.load_zoo_dataset(
    #     "coco-2017",
    #     splits=["train", "validation"],
    #     label_types=["detection", "segmentation"],
    #     classes=["person", "bicycle", "car", "motorcycle", "bus", "train", "truck"],
    # )

    # dataset_traffic_sign = foz.load_zoo_dataset(
    #     "coco-2017",
    #     splits=["train", "validation"],
    #     label_types=["detection", "segmentation"],
    #     classes=["traffic_light", "stop sign"],
    # )

    # train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # valid_loader = data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # trainer = pl.Trainer(gpus=[0], precision=16, limit_train_batches=0.5)
    # trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
