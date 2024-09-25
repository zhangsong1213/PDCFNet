import math
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                Mish(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                nn.Sigmoid()  ## hardsigmoid
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1).cuda()
        return x * y

class CDConv2d(nn.Module):  ## 像素差分卷积
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(CDConv2d, self).__init__()
        assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
        assert kernel_size == 3, 'kernel size for cd_conv should be 3x3'
        assert padding == dilation, 'padding for cd_conv set wrong'
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weights_c = self.weight.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(x, weights_c, stride=self.stride, padding=0, groups=self.groups)
        y = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y - yc


class ADConv2d(nn.Module):  ## 像素差分卷积
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ADConv2d, self).__init__()
        assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
        assert kernel_size == 3, 'kernel size for ad_conv should be 3x3'
        assert padding == dilation, 'padding for ad_conv set wrong'
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        shape = self.weight.shape
        weights = self.weight.view(shape[0], shape[1], -1)
        weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
        y = F.conv2d(x, weights_conv, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y


class RDConv2d(nn.Module):   ## 像素差分卷积
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(RDConv2d, self).__init__()
        assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
        assert kernel_size == 3, 'kernel size for rd_conv should be 3x3'
        self.stride = stride
        self.padding = 2 * dilation
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        shape = self.weight.shape
        if self.weight.is_cuda:
            buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
        else:
            buffer = torch.zeros(shape[0], shape[1], 5 * 5)
        weights = self.weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        y = F.conv2d(x, buffer, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y


class DEModule(nn.Module):  ###  细节增强模块
    def __init__(self, dim):  ## dim=input dim
        super(DEModule, self).__init__()  # 添加这行
        self.Ref = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(dim, dim//4, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(dim, dim//4, 1, 1, 0)
        self.conv3 = torch.nn.Conv2d(dim, dim//2, 1, 1, 0)
        self.conv4 = torch.nn.Conv2d(dim, dim//2, 3, 1, 1)

        self.CDConv2d = CDConv2d(dim//4, dim//4, 3, padding=1, dilation=1)   ## (in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.ADConv2d = ADConv2d(dim//4, dim//4, 3, padding=1, dilation=1)
        self.RDConv2d = RDConv2d(dim//4, dim//4, 3, padding=1, dilation=1)
        self.act = Mish()

    def forward(self, x):
        conv_x = self.act(self.conv1(x))
        CDConv2d_x = self.act(self.CDConv2d(self.conv1(x)))
        ADConv2d_x = self.act(self.ADConv2d(self.conv1(x)))
        RDConv2d_x = self.act(self.RDConv2d(self.conv1(x)))
        out = torch.cat((conv_x, CDConv2d_x, ADConv2d_x, RDConv2d_x), 1)
        out = self.act(self.conv3(out))

        x = self.act(self.conv4(x))

        out1 = torch.cat((out, x), 1)

        return out1  ## 输出维度保持不变



class FFMM(nn.Module):  ###  特征融合模块
    def __init__(self, indim1, indim2, outdim):  ## 注意 x1 和x2的 顺序
        super(FFMM, self).__init__()  # 添加这行
        # self.MAM = MAM(indim1+indim2, indim1+indim2)
        # self.MAM2 = MAM(outdim*3, outdim*3)
        self.MAM = SELayer(indim1+indim2, indim1+indim2)
        self.MAM2 = SELayer(outdim*3, outdim*3)

        # self.Ref = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(indim1, outdim, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(indim2, outdim, 1, 1, 0)
        self.conv3 = torch.nn.Conv2d(indim1+indim2, outdim, 1, 1, 0)
        self.conv4 = torch.nn.Conv2d(outdim*3, outdim, 1, 1, 0)
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.MAM(x)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        x = self.conv3(x)
        x1 = x1 * x
        x2 = x2 * x

        out = torch.cat((x1, x2, x), 1)
        out = self.MAM2(out)
        out = self.conv4(out)
        return out

class XXNet(torch.nn.Module):  ### PDCNet
    def __init__(self):
        super().__init__()
        self.Ref = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(3, 128, 3, 1, 0)
        self.DEM1 = DEModule(128)
        self.conv2 = torch.nn.Conv2d(128, 256, 3, 1, 0)
        self.DEM2 = DEModule(256)
        self.conv3 = torch.nn.Conv2d(256, 128, 3, 1, 0)
        self.DEM3 = DEModule(128)
        self.conv4 = torch.nn.Conv2d(128, 3, 3, 1, 0)
        self.FFM1 = FFMM(128, 128, 128)
        self.FFM2 = FFMM(256, 128, 128)

        self.act = Mish()
        self.sigmoid = torch.nn.Sigmoid()

        self.BN1 = torch.nn.InstanceNorm2d(128)
        self.BN2 = torch.nn.InstanceNorm2d(256)
        self.BN3 = torch.nn.InstanceNorm2d(128)

    def forward(self, x):
        x = self.Ref(x)
        x1 = self.conv1(x)
        x1 = self.BN1(x1)  ## 新加的
        x1 = self.act(x1)  ## 新加的

        x_DEM1 = self.DEM1(x1)
        x_DEM1 = self.DEM1(x_DEM1)
        x_DEM1 = self.DEM1(x_DEM1)

        x2 = self.Ref(x_DEM1)
        x2 = self.conv2(x2)
        x2 = self.BN2(x2)  ## 新加的
        x2 = self.act(x2)  ## 新加的

        x_DEM2 = self.DEM2(x2)
        x_DEM2 = self.DEM2(x_DEM2)
        x_DEM2 = self.DEM2(x_DEM2)

        x3 = self.Ref(x_DEM2)
        x3 = self.conv3(x3)
        x3 = self.BN3(x3)  ## 新加的
        x3 = self.act(x3)  ## 新加的

        # x3 = torch.cat((x1, x3), 1)
        x3 = self.FFM1(x1, x3)

        x_DEM3 = self.DEM3(x3)
        x_DEM3 = self.DEM3(x_DEM3)
        x_DEM3 = self.DEM3(x_DEM3)

        # x4 = torch.cat((x2, x_DEM3), 1)
        x4 = self.FFM2(x2, x_DEM3)

        x4 = self.Ref(x4)
        out = self.conv4(x4)
        out = self.sigmoid(out)

        return out



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = XXNet().to(device)
# input_tensor = torch.randn(1, 3, 256, 256).to(device)
# output_tensor = model(input_tensor)
# print("Output shape:", output_tensor.shape)

