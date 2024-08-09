from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
from torch.nn import Conv2d

from .unet_model import UNet


def get_model(model_name):
    if model_name == 'fcn':
        net = fcn_resnet50(num_classes=1)
        net.backbone.conv1 = Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_name == 'unet':
        net = UNet(n_channels=1, n_classes=1)
    elif model_name == 'deeplab':
        net = deeplabv3_resnet50(num_classes=1)
        net.backbone.conv1 = Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return net
