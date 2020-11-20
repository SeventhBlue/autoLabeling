#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2020/11/5 9:43
# @Author      :weiz
# @ProjectName :autoLabeling
# @File        :parse_config.py
# @Description :
import os

def cfgRead(path):
    """
    读取yolo的配置文件
    :param path:
    :return:
    """
    if not os.path.exists(path):
        print("This [{}] file does not exist!".format(path))
        exit()

    cfgFile = open(path, 'r')
    parseMaps = []   # 模型参数，类型为字典

    for line in cfgFile.readlines():
        line = line.strip()                     # 去掉行 头尾 的空白符（空格、回车等）
        if not line or line.startswith('#'):    # 去掉注释
            continue
        if line.startswith('['):                # 读取模块的始点
            parseMaps.append({})
            parseMaps[-1]["type"] = line[1:-1].strip()
            if(parseMaps[-1]["type"] == "convolutional"):
                parseMaps[-1]["batch_normalize"] = 0
        else:
            key, value = line.split('=')
            parseMaps[-1][key.strip()] = value.strip()

    return parseMaps


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
