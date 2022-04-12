#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 18:06
# @Author  : shiman
# @File    : utils.py
# @describe:

import numpy as np



def get_classes(classes_path):
    with open(classes_path, 'a') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_anchors(anchors_path):
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()

    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1,2)
    return anchors, len(anchors)