#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 18:06
# @Author  : shiman
# @File    : utils.py
# @describe:

import numpy as np
from PIL import Image



def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_anchors(anchors_path):
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()

    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1,2)
    return anchors, len(anchors)

def cvtColor(image):
    if len(np.shape(image)) == 3 or np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_image(image, out_size, letterbox_image):
    iw, ih = image.size
    w, h = out_size
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', out_size, (128,128,128))
        new_image.paste(image, ((w-nw)//2,(h-nh)//2))
    else:
        new_image = image.resize((w,h), Image.BICUBIC)

    return new_image

def preprocess_input(image):
    image /= 255.0
    return image

