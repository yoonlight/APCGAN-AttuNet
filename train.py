import time

import matplotlib.pyplot as plt
import pandas as pd
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torch.nn import Conv2d

from model.unet_model import UNet
from model.enet import ENet
from model.Models import AttU_Net, AttU_Net_min
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50, lraspp_mobilenet_v3_large

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pytorch_grad_cam import GradCAM


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    print(isbi_dataset)
    per_epoch_num = len(isbi_dataset) / batch_size


    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    #target_layers = [net.down4.maxpool_conv[-1]]
    # writer = SummaryWriter('logs/minpool-logsigmoid')
    loss2=[]
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
                if loss1 < best_loss:
                    best_loss = loss1
                    torch.save(net.state_dict(), 'best_model.pth')
                # 更新参数
                loss.backward()
                optimizer.step()
                pbar.update(1)
            # writer.add_scalar('loss', loss1, epoch)
            loss2.append(loss1)
    # writer.close()
    endtime = time.time()


    dict = {'loss': loss2}
    dict = pd.DataFrame(dict)
    dict.to_csv('./results/train_loss.csv')
    print('trainingtime:', endtime - starttime)


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    # original net
    # net = UNet(n_channels=1, n_classes=1)  # todo edit input_channels n_classes
    # attention unet
    # net = AttU_Net(img_ch=1, output_ch=1)
    # FCN
    # net = fcn_resnet50(num_classes=1)
#-------------------------
    # net = deeplabv3_resnet50(num_classes= 1)

    # net = fcn_resnet50(num_classes=1)
    net = ENet(num_classes=1,in_channels=1)


    # net = lraspp_mobilenet_v3_large(num_classes=1)
    # #
    # # # net.classifier._modules['6'] = nn.Linear(4096, 4)#for vgg16, alexnet
    # net.backbone.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # for vgg16, alexnet
    #
    #net.backbone._modules['0']._modules['0'] = Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


#--------------------------
    # net.to(device)



    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "./images/cracks" # todo 修改为你本地的数据集位置
    train_net(net, device, data_path, epochs=300, batch_size=16)
