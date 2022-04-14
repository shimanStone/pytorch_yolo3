#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/14 13:36
# @Author  : shiman
# @File    : yolo_training.py
# @describe:

import math
import torch.nn as nn
from functools import partial

def weight_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and class_name.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif class_name.find('BatchNorm2d') !=-1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        print(f'initialize network with {init_type}')

    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1,
                     warmup_lr_ratio=0.1, no_aug_iter_ratio=0.3, step_num=10):
    
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, 
                          warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start)*pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5*(lr - min_lr) * (
                1.0+math.cos(math.pi*(iters-warmup_total_iters)/(total_iters - warmup_total_iters - no_aug_iter))

            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError('step_size must above 1')

        n = iters // step_size
        out_lr = lr*decay_rate**n
        return out_lr
    
    if lr_decay_type == 'cos':
        warmup_total_iters = min(max(warmup_iters_ratio*total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio*lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio*total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr/lr) ** (1/(step_num-1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, Cuda, anchors_mask=[[6,7,8],[3,4,5],[0,1,2]]):
        super(YOLOLoss, self).__init__()
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        self.anchors = anchors
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.giou = True
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0]*input_shape[1]) / 416**2
        self.cls_ratio = 1 * (num_classes / 80)

        self.ignore_threshold = 0.5
        self.cuda = Cuda