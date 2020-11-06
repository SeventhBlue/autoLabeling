#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2020/11/5 9:41
# @Author      :weiz
# @ProjectName :autoLabeling
# @File        :augmentations.py
# @Description :
import torch


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

