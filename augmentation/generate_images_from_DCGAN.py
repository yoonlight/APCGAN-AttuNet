# import os
#
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
#
#
# class myDataset(Dataset):
#     def __init__(self, data_dir, transform):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.img_names = os.listdir(self.data_dir)
#
#     def __getitem__(self, index):
#         path_img = os.path.join(self.data_dir, self.img_names[index])
#         img = Image.open(path_img).convert('RGB')
#
#         if self.transform is not None:
#             img = self.transform(img)
#         return img
#
#     def __len__(self):
#         return len(self.img_names)
#
# data_dir = "C:/Users/tjzhang/Documents/TJzhang/gan_for_crack/data/cracks/transverse"
#
#
# mydata = myDataset(data_dir, transforms.ToTensor())
# img,label=mydata[1]
import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--data_dir", type=str, default='./data/deepcrack')
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

model = torch.load('./savemodel/DCGAN/DCGAN_G_6000.pth')
model.eval()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for i in range(276,300,1):
    # print('./images/'+str(i)+'.jpg')

    z = Variable(Tensor(np.random.normal(0, 1, (1, 100))))

    # Generate a batch of images
    gen_imgs = model(z)
    gen_imgs=gen_imgs.reshape([1,256,256])

    # print(type(gen_imgs), gen_imgs.shape)
    gen_imgs=gen_imgs.permute(1,2,0)
    plt.imshow(gen_imgs.cpu().detach().numpy(),'gray')
    plt.axis('off')
    # plt.imshow(
    #plt.show()
    plt.savefig('./augment_images/DCGAN/'+str(i)+'DCGAN.jpg',dpi=600,bbox_inches = 'tight',pad_inches=0.0)
    # print(gen_imgs)
    plt.clf()
    plt.close()

