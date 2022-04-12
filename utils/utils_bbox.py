#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 16:13
# @Author  : shiman
# @File    : utils_bbox.py
# @describe:

import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape,
                 anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]):
        super(DecodeBox, self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5+num_classes
        self.input_shape = input_shape
        # 13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        # 26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        # 52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        self.anchor_mask = anchor_mask

    def decode_box(self, inputs):
        outputs = []
        for i , input in enumerate(inputs):
            batch_size, _, input_height, input_width = input.size()
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            # 获得相对于特征层的scaled_anchor
            scaled_anchors = [(anchor_width/stride_w, anchor_height/stride_h) for anchor_width, anchor_height in self.anchors[self.anchor_mask[i]]]

            prediction = input.view(batch_size,len(self.anchor_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0,1,3,4,2).contigous()
            # 先验框中心位置调整
            x = torch.sigmoid(prediction[...,0])
            y = torch.sigmoid(prediction[...,1])
            w, h = prediction[...,2], prediction[...,3]
            # 是否有物体置信度
            conf = torch.sigmoid(prediction[...,4])
            # 种类置信度
            pred_cls = torch.sigmoid(prediction[...,5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # 生成先验框网格中心(bs,3,13,13)
            grid_x = torch.linspace(0, input_width-1, input_width).repeat(input_height,1).repeat(
                batch_size*len(self.anchor_mask[i]),1,1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height-1, input_height).repeat(input_width,1).t().repeat(
                batch_size*len(self.anchor_mask[i]),1,1).view(y.shape).type(FloatTensor)

            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor[1])
            anchor_w = anchor_w.repeat(batch_size,1).repeat(1,1,input_height*input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size,1).repeat(1,1,input_height*input_width).view(h.shape)

            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            #   将输出结果归一化成小数的形式
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)

        return  outputs



