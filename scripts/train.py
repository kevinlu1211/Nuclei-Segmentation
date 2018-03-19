import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.dataset import create_dataloader
from utils.preprocessing import create_data, load_data, train_val_split
from utils.evaluation import calculate_thresholded_precision, create_thresholded_mask
from utils.cuda import cudarize
from model import UNet
import json
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--config_file_path", default="config.json")
    opts = parser.parse_args()
    return opts


def main(imgs, masks, save_path, n_epochs, batch_size):

    train_imgs, train_masks, val_imgs, val_masks = train_val_split(imgs, masks)
    train_dl = create_dataloader(train_imgs, train_masks, is_train=True, batch_size=batch_size)
    val_dl = create_dataloader(val_imgs, val_masks, is_train=False, batch_size=batch_size)
    model = cudarize(UNet(n_input_channels=4, n_classes=2))
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
        for idx, batch in enumerate(tqdm(train_dl)):
            model.train()
            loss, precision, _ = step(model, optimizer, criterion, batch, is_train=True)
            tqdm.write(f"Precision: {round(precision.data[0], 3)} Loss: {round(loss.data[0], 3)}")

        val_loss, val_precision = [], []
        for idx, batch in enumerate(tqdm(val_dl)):
            model.eval()
            loss, precision, _ = step(model, optimizer, criterion, batch, is_train=False)
            tqdm.write(f"Precision: {round(precision.data[0], 3)} Loss: {round(loss.data[0], 3)}")
            val_loss.append(loss)
            val_precision.append(val_precision)

        val_avg_loss = np.mean(val_loss)
        val_avg_precision = np.mean(val_precision)
        epoch_save_path = Path(f"{save_path}/{time.strftime('%Y%m%d-%H%M%S')}/epoch_{epoch}.precision_{val_avg_precision}.loss_{val_avg_loss}")
        epoch_save_path.mkdir(parents=True, exist_ok=True)



def step(model, optimizer, criterion, batch, is_train):
    thresholds = range(50, 100, 5)
    inp = cudarize(Variable(batch['input']))
    pred = F.log_softmax(model(inp), dim=1)
    target = cudarize(Variable(batch['mask'])).unsqueeze(1).long()  # to keep same dims as pred
    # precision = calculate_thresholded_precision(pred.cpu(), target.cpu(), thresholds)
    # masks = create_thresholded_mask(pred.cpu(), thresholds)
    loss = criterion(pred, target.squeeze(1))
    if is_train:
        loss.backward()
        optimizer.step()
    precision = Variable(torch.rand(1))
    return loss, precision, masks

if __name__ == "__main__":
    opts = parse_arguments()
    with open(opts.config_file_path, "r") as fp:
        config = json.load(fp)

    if config["load_from_disk"]:
        imgs, masks = load_data(config["image_load_path"], config["mask_load_path"])
    else:
        imgs, masks = create_data(config["data_path"])

    main(imgs, masks, config["save_path"], config["n_epochs"], config["batch_size"])
