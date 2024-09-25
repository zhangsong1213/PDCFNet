import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class CDConv2d(nn.Module):
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


class ADConv2d(nn.Module):
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


class RDConv2d(nn.Module):
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


# # 示例输入张量
# x = torch.randn(1, 3, 32, 32)
#
# # 创建并调用 cdconv
# cdconv = CDConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
# output_cd = cdconv(x)
# print("cdconv output shape:", output_cd.shape)
#
# # 创建并调用 adconv
# adconv = ADConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
# output_ad = adconv(x)
# print("adconv output shape:", output_ad.shape)
#
# # 创建并调用 rdconv
# rdconv = RDConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
# output_rd = rdconv(x)
# print("rdconv output shape:", output_rd.shape)



