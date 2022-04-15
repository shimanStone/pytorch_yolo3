#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/15 10:38
# @Author  : shiman
# @File    : utils_fit.py
# @describe:

import os
import torch
from tqdm import tqdm

from ..nets.yolo_training import get_lr

def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                  gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):

    loss, val_loss = 0, 0


    print('start train')
    model_train.train()
    with tqdm(total=epoch_step, desc=f'epoch{epoch+1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration ,batch in enumerate(gen):
            if iteration > epoch_step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

            # 梯度清零
            optimizer.zero_grad()
            if not fp16:
                # 前向传播
                outputs = model_train(images)
                loss_value_all = 0
                # 计算在每个特征层下的损失函数
                for idx in range(len(outputs)):
                    loss_item = yolo_loss(idx, outputs[idx], targets)
                    loss_value_all +=loss_item
                loss_value = loss_value_all
                # 反向传播
                loss_value.backward()
                # 参数更新
                optimizer.step()
            loss +=loss_value.item()

            pbar.set_postfix(**{'loss':loss/(iteration+1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('Finish train')

    print('start val')
    model_train.eval()
    with tqdm(total=epoch_step_val, desc=f'epoch{epoch+1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #
            optimizer.zero_grad()
            #
            outputs = model_train(images)
            #
            loss_value_all = 0
            for idx in range(len(outputs)):
                loss_item = yolo_loss(idx, outputs[idx], targets)
                loss_value_all += loss_item
            loss_vale = loss_value_all
        val_loss += loss_vale.item()

        pbar.set_postfix(**{'val_loss':val_loss/(iteration+1)})
        pbar.update(1)
    print('Finish val')

    loss_history.append_loss(epoch+1, loss/epoch_step, val_loss/epoch_step_val)
    print(f'epoch:{epoch+1} / {Epoch}')
    print(f'total loss:{loss/epoch_step:.3f} || val loss:{val_loss/epoch_step_val}')
    if (epoch+1) % save_period == 0 or (epoch+1) == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, f'epoch{epoch+1:03d}-loss{loss/epoch_step,:.3f}-val_loss{val_loss/epoch_step_val:.3f}.pth'))



