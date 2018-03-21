import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.dataset import create_dataloader
from utils.preprocessing import create_data, load_data, train_val_split
# from utils.evaluation import calculate_IOU, create_thresholded_mask
from utils.cuda import cudarize
from utils.loss import soft_dice_loss
from model import UNet
import json
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time
from easydict import EasyDict as edict
import pickle

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--config_file_path", default="config.json")
    opts = parser.parse_args()
    return opts


def main(data, save_path, n_epochs, batch_size, criterion):

    train, val = train_val_split(data)
    train_dl = create_dataloader(train, is_test=False, batch_size=batch_size)
    val_dl = create_dataloader(val, is_test=False, batch_size=batch_size)
    model = cudarize(UNet(n_input_channels=3, n_classes=1))
    optimizer = torch.optim.Adam(model.parameters())
    start_time = time.strftime('%Y%m%d-%H%M%S')
    for epoch in range(n_epochs):
        for idx, batch in enumerate(tqdm(train_dl)):
            model.train()
            loss = step(model, optimizer, criterion, batch, is_train=True)
            tqdm.write(f"Loss: {round(loss.data[0], 3)}")

        val_loss, val_iou = [], []
        for idx, batch in enumerate(tqdm(val_dl)):
            model.eval()
            loss = step(model, optimizer, criterion, batch, is_train=False)
            tqdm.write(f"Loss: {round(loss.data[0], 3)}")
            val_loss.append(loss.cpu().data[0])

        val_avg_loss = round(np.mean(val_loss), 3)
        epoch_save_path = Path(f"{save_path}/{start_time}")
        epoch_save_path.mkdir(parents=True, exist_ok=True)
        checkpoint_save_name = f"epoch_{epoch}.loss_{val_avg_loss}.pth"
        torch.save(model.state_dict(), f"{epoch_save_path}/{checkpoint_save_name}")


def step(model, optimizer, criterion, batch, is_train):
    inp = cudarize(Variable(batch['input']))
    pred = F.sigmoid(model(inp))
    target = cudarize(Variable(batch['mask'])).unsqueeze(1)  # to keep same dims as pred
    loss = criterion(pred, target)

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

if __name__ == "__main__":
    opts = parse_arguments()
    with open(opts.config_file_path, "r") as fp:
        config = edict(json.load(fp))

    if config.load_from_disk:
        data = load_data(config.data_save_path)
    else:
        data = create_data(config.data_path)
        pickle.dump(data, open(config.data_save_path, "wb"))

    if config.criterion == "soft_dice_loss":
        criterion = soft_dice_loss
    else:
        criterion = nn.BCELoss()

    main(data, config.save_path, config.n_epochs, config.batch_size, criterion)
