import time

import matplotlib.pyplot as plt
import pandas as pd
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torch.nn import Conv2d

from model.unet_model import UNet
from model.enet import ENet
from model.Models import AttU_Net, AttU_Net_min
from utils.dataset_deepcrack import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50, lraspp_mobilenet_v3_large

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pytorch_grad_cam import GradCAM
from test_unet_deepcrack import cal_miou
from torch.utils.data import DataLoader


def train_net(net, device, data_path, val_data_path, result_path, dataset, expr_name, epochs=40, batch_size=1, lr=0.00001):
    model_file = f'{expr_name}_{dataset}_best_model.pth'

    # 加载训练集
    val_dataset = ISBI_Loader(val_data_path)
    isbi_dataset = ISBI_Loader(data_path)
    print(isbi_dataset)
    per_epoch_num = len(isbi_dataset) / batch_size


    train_loader = DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = DataLoader(dataset=val_dataset)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    #target_layers = [net.down4.maxpool_conv[-1]]
    # writer = SummaryWriter('logs/minpool-logsigmoid')
    loss2=[]
    val_loss_list = []
    starttime=time.time()
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            # 训练模式
            net.train()
            # 按照batch_size开始训练
            loss1=0
            for image, label in train_loader:
                optimizer.zero_grad()
                #plt.imshow(np.transpose(image.cpu()[1], (1, 2, 0)), interpolation='nearest',cmap='gray')
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                #plt.imshow(np.transpose(image.cpu()[1], (1, 2, 0)), interpolation='nearest', cmap='gray')

                # print(image.size())
                # plt.imshow(image.cpu()[1][0],cmap='gray')
                #plt.show()

                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = net(image)

                # 计算loss
                #loss = criterion(pred['out'], label)
                loss = criterion(pred, label)
                #losses.append(loss)
                loss1=loss1+loss.item()
                # writer.add_images('raw_images',image, epoch)
                # writer.add_images('pred_images', pred, epoch)
                #
                # writer.add_images('labeled_images', label, epoch)

                #print(losses)
                # print('{}/{}：Loss/train'.format(epoch + 1, epochs), loss.item())
                # 保存loss值最小的网络参数

                # 更新参数
                loss.backward()
                optimizer.step()
                pbar.update(1)
            # writer.add_scalar('loss', loss1, epoch)

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for image, label in val_loader:
                    image = image.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)
                    pred = net(image)
                    loss = criterion(pred, label)
                    val_loss += loss.item()

            val_loss_list.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), model_file)

            test_dir = f"datasets/seg/{dataset}/test/images"
            pred_dir = f"datasets/seg/{dataset}/{expr_name}/{expr_name}_images"
            gt_dir = f"datasets/seg/{dataset}/test/masks"
            miou_out_path = f"datasets/seg/{dataset}/{expr_name}/results/"
            if epoch % 10 == 0:
                cal_miou(test_dir, pred_dir, gt_dir, miou_out_path, model_file)
            pbar.set_postfix(loss=loss1, val_loss=val_loss)
            loss2.append(loss1)
    # writer.close()
    endtime = time.time()


    dict = {'train_loss': loss2, 'val_loss': val_loss_list}
    dict = pd.DataFrame(dict)
    dict.to_csv(f'{result_path}/train_loss.csv')
    print('trainingtime:', endtime - starttime)


if __name__ == "__main__":
    import os
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Choose a deep learning method:

    

    1. net = AttU_Net(img_ch=1, output_ch=1)

    2. net = AttU_Net_min(img_ch=1, output_ch=1)
    
    3. net = UNet(n_channels=1, n_classes=1)

    4. net = fcn_resnet50(num_classes=1)

    5.  net = deeplabv3_resnet50(num_classes= 1)

    6. net = lraspp_mobilenet_v3_large(num_classes=1)



    '''
    # 加载网络，图片单通道1，分类为1。
    # original net
    net = UNet(n_channels=1, n_classes=1)  # todo edit input_channels n_classes
    # attention unet
    # net = AttU_Net(img_ch=1, output_ch=1)
    # FCN
    # net = fcn_resnet50(num_classes=1)
#-------------------------
    # net = deeplabv3_resnet50(num_classes= 1)

    # net = fcn_resnet50(num_classes=1)
    # net = AttU_Net(img_ch=1, output_ch=1)


    # net = lraspp_mobilenet_v3_large(num_classes=1)
    # #
    # # # net.classifier._modules['6'] = nn.Linear(4096, 4)#for vgg16, alexnet
    # net.backbone.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # for vgg16, alexnet
    #
    #net.backbone._modules['0']._modules['0'] = Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


#--------------------------
    # net.to(device)
    net = nn.DataParallel(net, device_ids=[0, 1])


    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    # data_path = "./images/cracks" # todo: your training datasets
    expr_name = "unet-1"
    dataset = "DeepCrack"
    data_path = f"datasets/seg/{dataset}/train" # todo: your training datasets
    val_data_path = f"datasets/seg/{dataset}/test" # todo: your training datasets
    '''
    four choices: 
    "./images/cracks"
    "./images/cracks_tradition"
    "./images/cracks_DCGAN"
    "./images/cracks_APCGAN"
    '''
    result_path = f'datasets/seg/{dataset}/{expr_name}/results'
    os.makedirs(result_path, exist_ok=True)
    train_net(net, device, data_path, val_data_path, result_path, dataset, expr_name, epochs=300, batch_size=32, lr=1e-4)
