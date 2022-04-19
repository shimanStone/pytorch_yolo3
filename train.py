#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/14 10:54
# @Author  : shiman
# @File    : train.py
# @describe:

import os
import warnings
warnings.filterwarnings("ignore")

import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)
print(f'cur_dir: {cur_dir}')

sys.path.append(os.path.abspath(cur_dir))
sys.path.append(os.path.abspath(root_dir))
print(sys.path)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from pytorch_yolo3.utils.utils import get_classes, get_anchors
from pytorch_yolo3.nets.yolo import YoloBody
from pytorch_yolo3.nets.yolo_training import weight_init, YOLOLoss, get_lr_scheduler, set_optimizer_lr
from pytorch_yolo3.utils.callbacks import LossHistory
from pytorch_yolo3.utils.dataloader import YoloDataset, yolo_dataset_collate
from pytorch_yolo3.utils.utils_fit import fit_one_epoch


if __name__ == '__main__':
    Cuda = True if torch.cuda.is_available() else False
    fp16 = False # 是否混合精度训练
    #
    classes_path = f'{cur_dir}/data/voc_classes.txt'
    #
    anchor_path = f'{cur_dir}/data/yolo_anchors.txt'
    anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]
    #
    model_path = f'{cur_dir}/data/yolo_weights_coco2017.pth'
    input_shape = [416,416]  # 32的倍数

    # 是否使用主干网络预训练权重，model_path != '',则该参数无效
    pretrained = False

    # 冻结训练
    init_epoch, freeze_epoch, freeze_batch_size = 0, 50, 16
    # 解冻训练
    unfreeze_epoch, unfreeze_batch_size = 100, 8
    # 是否进行冻结训练
    freeze_train = True

    # 学习率
    init_lr = 1e-2
    min_lr = init_lr*0.01

    # 优化器
    optimizer_type = 'sgd'
    momentum = 0.937
    weight_decay = 5e-4
    # 学习率下降方法
    lr_decay_type = 'cos'  # cos / step
    # 多少个epoch保存一次权重
    save_period = 4
    #
    save_dir = 'logs'
    #  多线程
    num_workers = 2
    #
    # 图片和标签路径
    if Cuda:
        train_annotation_path, val_annotation_path = '2007_train_colab.txt', '2007_val_colab.txt'
    else:
        train_annotation_path, val_annotation_path = '2007_train.txt', '2007_val.txt'
    # train_annotation_path, val_annotation_path = '2007_train.txt', '2007_val.txt'
    # 获取分类信息和anchor
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchor_path)
    #
    model = YoloBody(anchor_mask, num_classes, pretrained=pretrained)
    if not pretrained:
        weight_init(model)
    if os.path.exists(model_path):
        print(f'load weights {model_path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # 定义损失函数
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchor_mask)
    loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    if fp16:
        from torch.cuda.amp import GradScaler as GardScaler
        scaler = GardScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # 读取dataset
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)

    #
    if True:
        unfreeze_flag = False

        if freeze_train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size
        #
        nbs = 64
        lr_limit_max = 0.001 if optimizer_type == 'adam' else 0.05
        lr_limit_min = 0.0003 if optimizer_type == 'adam' else 0.0005
        init_lr_fit = min(max(batch_size/nbs * init_lr, lr_limit_min),lr_limit_max)
        min_lr_fit = min(max(batch_size/nbs * min_lr, lr_limit_min*1e-2), lr_limit_max*1e-2)
        # 优化器
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                pg0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adma': optim.Adam(pg0, init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({'params':pg1, 'weight_decay': weight_decay})
        optimizer.add_param_group({'params':pg2})
        # 学习率
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unfreeze_epoch)
        #
        epoch_step, epoch_step_val = num_train // batch_size, num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('dataset numbers 过小，请扩充数据')

        # 构建数据集加载器
        train_dataset = YoloDataset(train_lines, input_shape, True)
        val_dataset = YoloDataset(val_lines, input_shape, False)
        #
        gen =  DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        # 开始进行模型训练
        for epoch in range(init_epoch, unfreeze_epoch):

            if epoch < freeze_epoch:

                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
                fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                              gen, gen_val, unfreeze_epoch, Cuda, fp16, scaler, save_period, save_dir, 0)
            else:
                batch_size = unfreeze_batch_size
                # 根据batch_size 自适应调整学习率
                nbs = 64
                lr_limit_max = 0.001 if optimizer_type == 'adam' else 0.05
                lr_limit_min = 0.0003 if optimizer_type == 'adam' else 0.0005
                init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
                min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unfreeze_epoch)
                #
                epoch_step, epoch_step_val = num_train // batch_size, num_val // batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError('dataset numbers 过小，请扩充数据')
                #
                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
                #
                for param in model.backbone.parameters():
                    param.requires_grad = True
                #
                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
                fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                              gen, gen_val, unfreeze_epoch, Cuda, fp16, scaler, save_period, save_dir, 0)

        loss_history.writer.close()