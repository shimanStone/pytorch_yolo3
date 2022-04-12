#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 14:36
# @Author  : shiman
# @File    : yolo.py
# @describe:

import torch
import torch.nn as nn
from .darknet import darknet53

from collections import OrderedDict

def conv2d(in_filter, out_filter, kernel_size):
    pad = (kernel_size - 1)//2 if kernel_size else 0
    return nn.Sequential(OrderedDict[
        ('conv', nn.Conv2d(in_filter, out_filter, kernel_size=kernel_size, padding=pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_filter)),
        ('relu', nn.LeakyReLU(0.1)),
    ])


def make_last_layers(filters_list, in_filter, out_filter):
    """
    共七个卷积，前五个用于特征提取，后两个用于yolo网络预测结果
    """
    m = nn.Sequential(
        conv2d(in_filter, filters_list[0],1),
        conv2d(filters_list[0], filters_list[1],3),
        conv2d(filters_list[1], filters_list[0],1),
        conv2d(filters_list[0], filters_list[1],3),
        conv2d(filters_list[1], filters_list[0],1),
        conv2d(filters_list[0], filters_list[1],3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )

    return m


class YoloBody(nn.Module):

    def __init__(self,anchor_mask, num_classes, pretrained=False):
        super(YoloBody, self).__init__()

        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load('../../data/yolo/darknet53_backbone_weight.pth'))

        out_filters = self.backbone.layers_outer_filter
        # 计算yolo_head的输出通道数
        # final_out_filters0 = final_out_filters1 =final_out_filters2 = 75
        self.last_layer0 = make_last_layers([512,1024], out_filters[-1], len(anchor_mask[0])*(num_classes+5))

        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2]+256, len(anchor_mask[1])*(num_classes+5))

        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3]+128, len(anchor_mask[2])*(num_classes+5))

    def forward(self, x):

        # 主干特征网络获取三个有效特征层
        x2, x1, x0 = self.backbone(x)

        out0_branch = self.last_layer0[:5](x0)
        out0 = self.last_layer0[5:](out0_branch)

        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], dim=1)

        out1_branch = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_branch)

        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], dim=1)

        out2 = self.last_layer2(x2_in)

        return out0, out1, out2

