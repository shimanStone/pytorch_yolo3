#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/14 15:55
# @Author  : shiman
# @File    : dataloader.py
# @describe:

import cv2
import numpy as np

from PIL import Image
from torch.utils.data.dataset import Dataset
from .utils import cvtColor, preprocess_input


class YoloDataset(Dataset):

    def __init__(self, annotation_lines, input_shape, train):
        super(YoloDataset, self).__init__()
        self.annotion_lines = annotation_lines
        self.input_shape = input_shape
        self.length = len(annotation_lines)
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        image: 经过变化resize后的值归一化的影像
        box: 经过归一化后的 c_x, c_y, w, h
        """
        index = index % self.length

        image, box = self.get_random_data(self.annotion_lines[index], self.input_shape[0:2], random=self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2,0,1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:,[0,2]] = box[:,[0,2]] / self.input_shape[1]
            box[:,[1,3]] = box[:,[1,3]] / self.input_shape[0]

            box[:,[2,3]] = box[:,[2,3]] - box[:,[0,1]]
            box[:,[0,1]] = box[:,[0,1]] + box[:,[2,3]] /2
        return image, box


    def rand(self, a=0, b=1):
        # b>a时, c<b
        # b<a时, c>b
        c = np.random.rand()*(b-a) + a
        return c

    def get_random_data(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7,
                        val=0.4, random=True):
        line = annotation_line.split()
        # 数据读取转换
        image = Image.open(line[0])
        image = cvtColor(image)
        #
        iw, ih = image.size
        h, w = input_shape
        #
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        #
        if not random:
            scale = min(w/iw, h/ih)
            nw, nh = int(iw*scale), int(ih*scale)
            dx = (w-nh) // 2
            dy = (h-nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx,dy))
            image_data = np.array(new_image, dtype='float32')

            # 对真实框进行调整
            if len(box) > 0:
                np.random.shuffle(box)
                box[:,[0,2]] = box[:,[0,2]] * scale + dx
                box[:,[1,3]] = box[:,[1,3]] * scale + dy
                box[:,[0,1]][box[:,[0,1]]<0] = 0
                box[:,2][box[:,2] > w] = w
                box[:,3][box[0,3] > h] = h
                box_w = box[:,2] - box[:,0]
                box_h = box[:,3] - box[:,1]
                box = box[np.logical_and(box_w>1, box_h>1)]
            return image_data, box
        # 对图像进行处理(缩放，长宽扭曲)
        new_ar = iw/ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(.25, 2)  # (0,2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)  # nh/ih * iw
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar) # nw/iw * ih
        image = image.resize((nw,nh), Image.BICUBIC)
        #
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # 左右翻转
        flip = self.rand() < 0.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 色域变化
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1,1,3) *[hue, sat, val] + 1
        hue, sat ,val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))  # H = [0,179]；S = [0,255]；V = [0,255]
        dtype = image_data.dtype
        #
        x = np.arange(0,256,dtype=r.dtype)
        lut_hue = ((x*r[0])%180).astype(dtype)
        lut_sat = np.clip(x*r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x*r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # 对真实框进行调整
        if len(box) > 0:
            box[:,[0,2]] = box[:,[0,2]] * (nw/iw) + dx
            box[:,[1,3]] = box[:,[1,3]] * (nh/iw) + dy
            if flip: box[:,[0,2]] = w - box[:,[2,0]]
            box[:,[0,1]][box[:,[0,1]] < 0] = 0
            box[:,2][box[:,2] > w] = w
            box[:,3][box[:,3] > h] = h
            box_w = box[:,2] - box[:,0]
            box_h = box[:,3] - box[:,1]
            box = box[np.logical_and(box_w >1, box_h>1)]
        return image_data, box

def yolo_dataset_collate(batch):
    images, box = [], []
    for i, b in batch:
        images.append(i)
        box.append(b)
    images = np.array(images)
    return images, box

