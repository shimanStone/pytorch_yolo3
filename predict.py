#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 17:17
# @Author  : shiman
# @File    : predict.py
# @describe:


from PIL import Image

from pytorch_yolo3.nets.yolo_detection import YOLO


if __name__ == '__main__':

    yolo = YOLO()

    mode = 'predict'
    crop=False

    img = r'E:\ml_code\data\frcnn\VOC2007\JPEGImages\000048.jpg'
    image = Image.open(img)
    r_image = yolo.detect_image(image, crop=crop)
    r_image.show()
