import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor
from utils1 import  ResBlock, ConvBlock, Up, Compute_z

import torch.nn as nn
from torch.distributions import Normal, Independent

# import numpy as np
# from PIL import Image
from scipy.ndimage import minimum_filter  # 导入minimum_filter函数
from torchvision.transforms.functional import to_tensor, to_pil_image



def my_save_image(name, image_np, output_path=""):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    p = np_to_pil(image_np)
    p.save(output_path + "{}".format(name))


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8) ## 还原像素值

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def get_A(x): ## 本身输入的x的数值就是 0-1

    x_np = np.clip(torch_to_np(x), 0, 1)  ##
    x_pil = np_to_pil(x_np)  ### 还原到255
    h, w = x_pil.size
    windows = (h + w) / 2
    A = x_pil.filter(ImageFilter.GaussianBlur(windows))
    A = ToTensor()(A)  ## 归一化0-1
    return A.unsqueeze(0)


#
# def get_A(x):
#     x_np = np.clip(torch_to_np(x), 0, 1)
#     x_pil = np_to_pil(x_np)
#     h, w = x_pil.size
#     imsz = h * w
#     # 要查找的是暗通道中前0.1%的值
#     numpx = max(imsz // 1000, 1/255)
#     # 找到暗通道的索引，弄成[batch, 3, numpx]，因为要匹配三个通道，所以需要expand
#     dark = torch.min(x, dim=1, keepdim=True)[0]
#     indices = torch.topk(dark.view(-1, imsz), k=numpx, dim=1)[1].view(-1, 1, numpx).expand(-1, 3, -1)
#     # 用上述索引匹配原图中的3个通道，并求其平均值
#     a = (torch.gather(x.view(-1, 3, imsz), 2, indices).sum(2) / numpx).unsqueeze(0)
#
#     return a


# def get_A(x):
#     x_np = np.clip(torch_to_np(x), 0, 1)
#     x_pil = np_to_pil(x_np)
#     h, w = x_pil.size
#     imsz = h * w
#     # 要查找的是暗通道中前0.1%的值
#     numpx = max(imsz // 1000, 1/255)
#     # 找到暗通道的索引，弄成[batch, 3, numpx]，因为要匹配三个通道，所以需要expand
#     dark = torch.min(x, dim=1, keepdim=True)[0]
#     indices = torch.topk(dark.view(-1, imsz), k=numpx, dim=1)[1].view(-1, 1, numpx).expand(-1, 3, -1)
#     # 用上述索引匹配原图中的3个通道，并求其平均值
#     a = (torch.gather(x.view(-1, 3, imsz), 2, indices).sum(2) / numpx).unsqueeze(0)
#
#     return a



