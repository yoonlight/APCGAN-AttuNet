# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: test.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
import os
import time

import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch import nn

from utils.utils_metrics import compute_mIoU, show_results
from model import get_model
import argparse


def get_image_and_label_paths(dataset_name, test_dir, gt_dir, image_id):
    if dataset_name == "BCL":
        image_path = os.path.join(test_dir, image_id + ".jpg")
        label_path = os.path.join(gt_dir, image_id + ".jpg")
    elif dataset_name == "DeepCrack":
        image_path = os.path.join(test_dir, image_id + ".jpg")
        label_path = os.path.join(gt_dir, image_id + ".png")
    elif dataset_name == "Volker":
        image_path = os.path.join(test_dir, image_id + ".png")
        label_path = os.path.join(gt_dir, image_id + ".png")
    else:
        image_path = os.path.join(test_dir, image_id + ".png")
        label_path = os.path.join(gt_dir, image_id + ".png")
    return image_path, label_path


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    origin_shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    img_tensor = torch.from_numpy(img).to(dtype=torch.float32)
    return img_tensor, origin_shape


def preprocess_label(label_path):
    label = cv2.imread(label_path)
    label = cv2.resize(label, (256, 256))
    label = (label / 255).astype(int)
    return label


def predict_image(net, img_tensor, device, model_name):
    img_tensor = img_tensor.to(device=device)
    with torch.no_grad():
        pred = net(img_tensor)
    if model_name in ['fcn', 'deeplab']:
        pred = pred['out']
    pred = np.array(pred.data.cpu()[0])[0]
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    return pred


def cal_miou(
    test_dir="./images/cracks/Test_Images",
    pred_dir="./images/cracks/results",
    gt_dir="./images/cracks/Test_Labels",
    model_name="unet",
    dataset_name="BCL",
    model_path="unet_BCL_best_model.pth",
    miou_out_path=""
):
    miou_mode = 0
    num_classes = 2
    name_classes = ["background", "crack"]

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = get_model(model_name)
        net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device=device)
        net.load_state_dict(torch.load(
            model_path, map_location=device))
        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)

        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        times = []
        for image_id in tqdm(image_ids):
            image_path, label_path = get_image_and_label_paths(
                dataset_name, test_dir, gt_dir, image_id)
            label = preprocess_label(label_path)
            img_tensor, origin_shape = preprocess_image(image_path)

            starttime = time.time()
            pred = predict_image(net, img_tensor, device, model_name)
            endtime = time.time()
            times.append(endtime-starttime)

            pred = cv2.resize(
                pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

        print("Get predict result done.")
        print(np.mean(times))

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes,
                                                        dataset_name=dataset_name
                                                        )  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs,
                     PA_Recall, Precision, name_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APCGAN-AttuNet')
    parser.add_argument('--iters', type=int, default=4000, help='number of iterations')
    args = parser.parse_args()

    EXPR = "final13"
    DATASET = "BCL_image"
    test_dir = f"datasets/stargan/{EXPR}/generate/{args.iters}/BCL_mask2{DATASET}/images"
    pred_dir = f"test/{EXPR}/{args.iters}/{DATASET}/pred"
    gt_dir = f"datasets/stargan/{EXPR}/generate/{args.iters}/BCL_mask2{DATASET}/origin"
    model_name = "fcn"
    model_path = "models/BCL/origin/fcn-1_best_model.pth"
    miou_out_path = f"test/{EXPR}/{args.iters}/{DATASET}"
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(miou_out_path, exist_ok=True)
    cal_miou(
        test_dir, pred_dir, gt_dir,
        model_name=model_name,
        dataset_name="",
        model_path=model_path,
        miou_out_path=miou_out_path
    )
