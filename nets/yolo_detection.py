#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 17:51
# @Author  : shiman
# @File    : yolo_detection.py
# @describe:

import colorsys
import torch
import torch.nn as nn
import numpy as np

from yolo3.utils.utils import get_classes, get_anchors, cvtColor, resize_image, preprocess_input, draw_detection_image
from yolo3.utils.utils_bbox import DecodeBox
from yolo3.nets.yolo import YoloBody

class YOLO(object):

    _defaults = {
        "model_path": ".././data/yolo/yolo_weights_coco2017.pth",
        'classes_path': '.././data/yolo/coco_classes.txt',
        'anchors_path': '.././data/yolo/yolo_anchors.txt',
        'anchors_mask': [[6,7,8],[3,4,5],[0,1,2]],
        'input_shape': [416,416],
        'confidence': 0.5,
        'nms_iou': 0.3,
        'letterbox_image': False, # 是否对输入图像进行不失真resize
        'cuda':False,
    }

    @classmethod
    def get_defaults(cls, k):
        if k in cls._defaults:
            return cls._defaults[k]
        else:
            return f'Unrecognized attribute name {k}'

    @classmethod
    def set_defaults(cls, k, v):
        if k in cls._defaults:
            cls._defaults[k] = v
        else:
            return f'Unrecognized attribute name {k}'


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for k, v in kwargs.items():
            setattr(self, k, v)
        # 获取种类和先验框
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes,
                                   (self.input_shape[0],self.input_shape[1]), self.anchors_mask)

        # 画框设置不同颜色
        hsv_tuples = [(x/self.num_classes,1,1) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), self.colors))
        self.generate()

    def generate(self):
        """生成模型，载入模型"""
        self.net = YoloBody(self.anchors_mask, self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print(f'{self.model_path} model, anchors and classes loaded')

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image, crop=False):

        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 添加batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2,0,1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 对图像进行预测
            outputs = self.net(images)
            # 解码
            outputs = self.bbox_util.decode_box(outputs)
            # 预测框堆叠进行非极大抑制
            result = self.bbox_util.non_max_suppression(torch.cat(outputs,1),self.num_classes, self.input_shape, image_shape,
                                                       self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
            if result[0] is None:
                return image

            # 绘制第一张图
            image = draw_detection_image(image, result[0], self.classes_path, self.input_shape)

            return image
















