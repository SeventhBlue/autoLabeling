#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2020/11/5 9:32
# @Author      :weiz
# @ProjectName :autoLabeling
# @File        :yoloMain.py
# @Description :yolo模型检测的入口
from __future__ import division
from torch.autograd import Variable
import cv2

from yolov3Models import *
from utils.utils import *
from utils.datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def getYolov3(modelPath=None, cfg=None, imgS=None):
    """
    获取yolov3模型
    :param yolov3ModelPath:
    :param yolov3Cfg:
    :param imgSize:
    :return:
    """
    if modelPath == None:
        modelPath = yolov3ModelPath
    if cfg == None:
        cfg = yolov3Cfg
    if imgS == None:
        imgS = imgSize

    # set up model
    model = Darknet(cfg, img_size=imgS).to(device)

    # Load darknet weights
    model.load_darknet_weights(modelPath)
    # Load checkpoint weights
    # model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode
    return model


def showResult(image, boxes, labels, confs, timeLabel):
    """
    显示结果
    :param boxes:
    :param labels:
    :param confs:
    :return:
    """
    for ind, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = "{} {:.2f}".format(labels[ind], confs[ind])
        cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(image, timeLabel, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return image


def runningYolov3(yolov3Model, image, cls=None):
    """
    运行yolov3模型
    :param yolov3Model:
    :param image:
    :param cls:
    :return:
    """
    if cls == None:
        cls = classes

    time_start = cv2.getTickCount()
    # numpy to tensor
    tensorImage = transforms.ToTensor()(image)
    # Pad to square resolution
    tensorImage, _ = pad_to_square(tensorImage, 0)
    # Resize
    tensorImage = resize(tensorImage, imgSize)

    # Configure input
    inputImage = Variable(torch.unsqueeze(tensorImage.type(Tensor), dim=0).float(), requires_grad = False)

    # Get detections
    with torch.no_grad():
        detections = yolov3Model(inputImage)
        detections = non_max_suppression(detections, configThres, nmsThres)

    if detections[0] is None:
        return [], [], [], "no object"

    boxes, labels, confs = rescale_boxes(detections, imgSize, image.shape[:2], cls)
    time_end = cv2.getTickCount()
    spend_time = (time_end - time_start) / cv2.getTickFrequency() * 1000
    fps = cv2.getTickFrequency() / (time_end - time_start)
    timeLabel = "Spend Time:{:.2f} FPS:{:.2f}".format(spend_time, fps)
    return boxes, labels, confs, timeLabel


yolov3ModelPath = "./cfg/yolov3_hsh_food.weights"      # "./cfg/yolov3.weights"
yolov3Cfg = "./cfg/yolov3_hsh_food.cfg"                # "./cfg/yolov3.cfg"
classes = load_classes("./cfg/yolov3_hsh_food.names")  # "./cfg/coco.names"
imgSize = 416
configThres = 0.6
nmsThres = 0.5
if __name__ == "__main__":
    img = cv2.imread("./srcImages/dog.jpg")
    yolov3 = getYolov3(yolov3ModelPath, yolov3Cfg, imgSize)
    cap = cv2.VideoCapture("./videos/004.avi")
    frameNum = 0
    gap = 1
    while True:
        ok, img = cap.read()
        frameNum = frameNum + 1
        if not ok:
            break
        if frameNum % gap != 0:
            continue
        boxes, labels, confs, timeLabel = runningYolov3(yolov3, img, classes)
        img = showResult(img, boxes, labels, confs, timeLabel)
        cv2.imshow('video', img)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()  # 关闭摄像头
            break