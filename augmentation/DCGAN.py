import argparse
import os
import numpy as np
import math

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--data_dir", type=str, default='./deepcrack/')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# -------------------------------load data--------------------------------------#

class myDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = os.listdir(self.data_dir)

    def __getitem__(self, index):
        path_img = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(path_img).convert('L')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_names)


transform = transforms.Compose(
    [transforms.Resize([opt.img_size,opt.img_size]),
     transforms.ToTensor()]
     #transforms.Normalize([0.5], [0.5])]
)


mydata = myDataset(opt.data_dir, transform)

dataloader = torch.utils.data.DataLoader(
    mydata,
    batch_size=opt.batch_size,
    shuffle=True)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
# writer = SummaryWriter('logs/DCGAN')
lossg2=[]
lossd2=[]
for epoch in range(opt.n_epochs):
    lossg1=0
    lossd1=0
    for i, imgs in enumerate(dataloader):
        # writer.add_images('real_images', imgs, epoch)

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)
        #
        # for k in range(32):
        #     imgg = Image.fromarray(np.uint8(im2[k] * 255))
        #     imgg.save("images/i" + str(epoch) + '-0-' + str(k) + ".png")

        # writer.add_images('generated_images', gen_imgs, epoch)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)

        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        lossg1 = lossg1 + g_loss.item()
        lossd1 = lossd1 + d_loss.item()

        # writer.add_scalar('Loss/Generator', lossg1, epoch)
        # writer.add_scalar('Loss/Discriminator', lossd1, epoch)

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        # )

        # batches_done = epoch * len(dataloader) + i
    if epoch % 1000 ==0:
        save_image(gen_imgs.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)
    # if batches_done % opt.sample_interval == 0:
    #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    if epoch % 1000 ==0:
        torch.save(generator, './savemodel/DCGAN/DCGAN_G_{}.pth'.format(epoch))
        # torch.save(discriminator, './savemodel/DCGAN/DCGAN_D_{}.pth'.format(epoch))
        print('The {}th model saved!'.format(epoch))
        print('G loss:{0},D loss:{1}'.format(lossg1, lossd1))

    lossg2.append(lossg1)
    lossd2.append(lossd1)

# writer.close()
dict = {'lossG': lossg2, 'lossD': lossd2}
dict = pd.DataFrame(dict)
dict.to_csv('./results/DCGAN/transverse.csv')
