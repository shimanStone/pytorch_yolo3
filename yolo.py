#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 17:51
# @Author  : shiman
# @File    : yolo.py
# @describe:

import colorsys
from .utils.utils import get_classes, get_anchors
from .utils.utils_bbox import DecodeBox

class YOLO(object):

    _defaults = {
        "model_path": "../data/yolo/yolo_weight.pth",
        'classes_path': '../data/yolo/coco_classes.txt',
        'anchors_path': '../data/yolo/yolo_anchors.txt',
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
        hsv_tuples = [(x/self.num_classes,1,1) for x in self.num_classes]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[3]*255)), self.colors))
        self.generate()

    def generate(self):
        """生成模型"""
        pass












