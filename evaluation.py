#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/4/7 9:37
# @Author      :weiz
# @ProjectName :autoLabeling
# @File        :evaluation.py
# @Description :训练好模型后进行模型的评估
import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

import yoloMain


def IOU(rec1, rec2):
    """
    rec1:[x1,y1,x2,y2]  (top, left, bottom, right)
    rec2:[x1,y1,x2,y2]  (top, left, bottom, right)
    """
    areas1 = (rec1[3] - rec1[1]) * (rec1[2] - rec1[0])
    areas2 = (rec2[3] - rec2[1]) * (rec2[2] - rec2[0])
    left = max(rec1[1], rec2[1])
    right = min(rec1[3], rec2[3])
    top = max(rec1[0], rec2[0])
    bottom = min(rec1[2], rec2[2])
    w = max(0, right-left)
    h = max(0, bottom-top)
    return w * h / (areas2+areas1-w*h)


def findMaxIOU(boxes_gt, boxesList, labels, confs):
    """
    在预测框中寻找与真实框有最大IOU的框
    :param boxes_gt: 真实框
    :param boxesList: 模型预测的所有框
    :param labels:
    :param confs:
    :return:
    """
    maxIOU = -1
    ind = 0
    for i, (x1, y1, x2, y2) in enumerate(boxesList):
        iou = IOU(boxes_gt, (x1, y1, x2, y2))
        if iou > maxIOU:
            maxIOU = iou
            ind = i

    return maxIOU, boxesList[ind], labels[ind], confs[ind]


def evaluation(path, modelSign):
    """
    评估函数
    :param path:
    :param modelSign:
    :return:
    """
    statRet = {}   # 保存统计结果，结构是{label: [[预测正确的conf,], [预测错误的label, ]]}

    if modelSign == 3:
        yoloModel = yoloMain.getYolov3()
    elif modelSign == 4:
        yoloModel = yoloMain.getYolov4()
    else:
        print("仅仅支持yolov3和yolov4")
        return

    imagesListNames = os.listdir(os.path.join(path, "images"))
    for imageName in imagesListNames:
        imagePath = os.path.join(path, "images", imageName)
        xmlPath = os.path.join(path, "xml", os.path.splitext(imageName)[0]+".xml")
        img = cv2.imread(imagePath)

        if modelSign == 3:
            boxes, labels, confs, timeLabel = yoloMain.runningYolov3(yoloModel, img)
        elif modelSign == 4:
            boxes, labels, confs, timeLabel = yoloMain.runningYolov4(yoloModel, img)

        try:
            tree = ET.parse(xmlPath)
        except FileNotFoundError:
            print("No such file or directory:" + xmlPath)
            continue
        root = tree.getroot()  # 获取xml根节点

        for obj in root.iter('object'):
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            (x1_gt, y1_gt, x2_gt, y2_gt) = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                                            float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))

            iou, box, label, conf = findMaxIOU((x1_gt, y1_gt, x2_gt, y2_gt), boxes, labels, confs)

            if cls in statRet.keys():  # cls这个label在statRet字典中已经存在
                if (cls == label) and (iou > 0):  # 预测正确：预测的label和真实的cls相同，且IOU大于零
                    statRet[cls][0].append(float('%.4f' % conf))  # 将预测的conf值添加到结果中
                else:
                    statRet[cls][1].append(label)  # 预测不正确：将预测的错误的label添加到结果中
            else:
                statRet[cls] = [[], []]
                if (cls == label) and (iou > 0):
                    statRet[cls][0].append(float('%.4f' % conf))
                else:
                    statRet[cls][1].append(label)

    return statRet


def linePlot(X, Y):
    """
    线图
    :param X:
    :param Y:
    :return:
    """
    plt.plot(X, Y)
    plt.scatter(X, Y)

    plt.xticks(X)
    plt.xlabel("IOU")
    plt.ylabel("Accuracy")
    plt.title("Accurateness")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()


def barPlot(labels, labelNumT, labelNumMin, labelNumF):
    """
    画条形图
    :param labels:标签
    :param labelNumT:预测正确且IOU大于等于0.5情况下的个数
    :param labelNumMin:预测正确但是IOU小于0.5情况的个数
    :param labelNumF:预测不正确的个数
    :return:
    """
    fig, ax = plt.subplots()

    ax.bar(np.arange(len(labels)), labelNumT, 0.5, label="True", color='green')
    ax.bar(np.arange(len(labels)), labelNumMin, 0.5, label="Mid", color='blue')
    ax.bar(np.arange(len(labels)), labelNumF, 0.5, label="False", color='red')

    ax.set_ylabel('Number')
    ax.set_xlabel('label')
    ax.set_title('Count the accuracy of different labels')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend()

    for a, b in zip(np.arange(len(labels)), labelNumT):
        plt.text(a, b / 2, '%.0f' % b, ha='center', va='bottom', fontsize=7)

    for a, b, c in zip(np.arange(len(labels)), labelNumT, labelNumMin):
        plt.text(a, b + c / 2, '%.0f' % c, ha='center', va='bottom', fontsize=7)

    for a, b, c, d in zip(np.arange(len(labels)), labelNumT, labelNumMin, labelNumF):
        plt.text(a, b + c + d / 2, '%.0f' % d, ha='center', va='bottom', fontsize=7)

    plt.show()


def statDrawing(statRet):
    """
    将统计数据用图表显示
    :param statRet: 结构是{label: [[预测正确的conf, ], [预测错误的label, ]]}
    :return:
    """
    polyLineX = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    polyLineY = []

    # 下面是统计准确性的数据
    for iouX in polyLineX:
        numT = 0  # 预测正确的个数
        num = 0   # 总共的个数
        for label in statRet.keys():  # 遍历所有的label
            for iou in statRet[label][0]:
                if iou >= iouX:
                    numT = numT + 1

            num = num + len(statRet[label][0]) + len(statRet[label][1])
        polyLineY.append(numT / num)
    linePlot(polyLineX, polyLineY)

    # 下面是统计每个标签在IOU大于0.5情况下准确性的数据
    labels = []       # 标签
    labelNumT = []    # 预测正确且IOU大于等于0.5情况下的个数
    labelNumF = []    # 预测不正确的个数
    labelNumMin = []  # 预测正确但是IOU小于0.5情况的个数
    for label in statRet.keys():
        labels.append(label)
        numT_tmp = 0
        numMid_tmp = 0
        for iou in statRet[label][0]:
            if iou >= 0.5:
                numT_tmp = numT_tmp + 1
            else:
                numMid_tmp = numMid_tmp + 1
        labelNumT.append(numT_tmp)
        labelNumF.append(len(statRet[label][1]))
        labelNumMin.append(numMid_tmp)
    barPlot(labels, labelNumT, labelNumMin, labelNumF)


evaluationPath = "./evaluation"                     # 评估的图片，包含两个文件夹，一个是images，一个xml
modelSign = 3                                       # 表示需要评估的模型：3表示评估yolov3，4表示评估yolov4
if __name__ == "__main__":
    statRet = evaluation(evaluationPath, modelSign)
    statDrawing(statRet)

