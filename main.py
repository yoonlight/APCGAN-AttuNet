import time
import argparse
import os

import pandas as pd
import torch

from tqdm import tqdm
from torch import nn
from torch import optim

from model import get_model
from utils import load_data
from main_test import cal_miou


def train_one_epoch(net, train_loader, optimizer, criterion, device, model_name):
    net.train()
    epoch_loss = 0
    for image, label in train_loader:
        optimizer.zero_grad()
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        pred = net(image)
        if model_name in ['fcn', 'deeplab']:
            pred = pred['out']
        loss = criterion(pred, label)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss


def validate_one_epoch(net, val_loader, criterion, device, model_name):
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            if model_name in ['fcn', 'deeplab']:
                pred = pred['out']
            loss = criterion(pred, label)
            val_loss += loss.item()
    return val_loss


def train_net(net, device, dataset_name, model_name, data_path, val_data_path, result_path, dataset, expr_name, epochs=40, batch_size=1, lr=0.00001):
    model_file = f'./models/{dataset}/{expr_name}_best_model.pth'
    test_dir = f"datasets/seg/{dataset}/test/images"
    pred_dir = f"datasets/seg/{dataset}/{expr_name}/{expr_name}_images"
    gt_dir = f"datasets/seg/{dataset}/test/masks"
    miou_out_path = f"datasets/seg/{dataset}/{expr_name}/results/"

    train_loader, val_loader, train_ds_len = load_data(
        dataset_name, data_path, val_data_path, batch_size)
    per_epoch_num = train_ds_len / batch_size

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')

    train_losses = []
    val_losses = []

    starttime = time.time()
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            train_loss = train_one_epoch(
                net, train_loader, optimizer, criterion, device, model_name)
            val_loss = validate_one_epoch(
                net, val_loader, criterion, device, model_name)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), model_file)

            if epoch % 10 == 0:
                cal_miou(test_dir, pred_dir, gt_dir, model_name=model_name,
                         dataset_name=dataset_name, model_path=model_file, miou_out_path=miou_out_path)
            pbar.set_postfix(loss=train_loss, val_loss=val_loss)
            pbar.update(per_epoch_num)
    endtime = time.time()

    loss_dict = {'train_loss': train_losses, 'val_loss': val_losses}
    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(f'{result_path}/train_loss.csv')
    print('trainingtime:', endtime - starttime)


def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--expr_name', type=str,
                        default='unet-1', help='Name of the experiment')
    parser.add_argument('--dataset_dir', type=str,
                        default='Volker/aug-6', help='Name of the dataset directory')
    parser.add_argument('--dataset', type=str,
                        default='Volker', help='Name of the dataset')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Size of each training batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_name', type=str,
                        default='fcn', help='Name of the model')
    return parser.parse_args()


def main():
    args = parse_args()

    net = get_model(args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.DataParallel(net, device_ids=[0, 1])
    net.to(device=device)

    data_path = f"datasets/seg/{args.dataset_dir}/train"
    val_data_path = f"datasets/seg/{args.dataset_dir}/test"
    result_path = f'datasets/seg/{args.dataset_dir}/{args.expr_name}/results'
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(f'models/{args.dataset_dir}', exist_ok=True)

    train_net(net, device, args.dataset, args.model_name, data_path, val_data_path, result_path,
              args.dataset_dir, args.expr_name, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
