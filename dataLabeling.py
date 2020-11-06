#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2020/11/3 14:00
# @Author      :weiz
# @ProjectName :autoLabeling
# @File        :dataLabeling.py
# @Description :数据标注文件
import os
import codecs
import cv2

import yoloMain


def vocXMLFormat(filePath, imgName, imgW, imgH, objNames, locs, depth=3, truncated=0, difficult=0):
    """
    生成xml数据文件;eg:
        vocXMLFormat("0000.xml", "0001.png", 608, 608, ["person", "TV"], [[24, 32, 156, 145], [124, 76, 472, 384]])
    :param filePath: 生成xml所保存文件的路径
    :param imgName: xml文件标注图片的名字
    :param imgW: 图片的宽
    :param imgH: 图片的高
    :param objNames: 图片包含的目标，格式：["目标1","目标2"...]
    :param locs: 图片包含目标的坐标，格式：[[x,y,w,h],[x,y,w,h]...]
    :param depth: 图片的深度，默认是3
    :param truncated: 是否被截断（0表示完整）
    :param difficult: 目标是否难以识别（0表示容易识别）
    :return:
    """
    if (objNames == None) or (locs == None):
        print("The objNames or locs is None!!!")
        return
    with codecs.open(filePath, 'w', 'utf-8') as xml:
        xml.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'voc format' + '</folder>\n')
        xml.write('\t<filename>' + imgName + '</filename>\n')
        xml.write('\t<path>' + imgName + '</path>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>weiz</database>\n')
        xml.write('\t</source>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(imgW) + '</width>\n')
        xml.write('\t\t<height>' + str(imgH) + '</height>\n')
        xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t<segmented>0</segmented>\n')

        for ind, name in enumerate(objNames):
            xml.write('\t<object>\n')
            xml.write('\t\t<name>' + name + '</name>\n')
            xml.write('\t\t<pose>Unspecified</pose>\n')
            xml.write('\t\t<truncated>' + str(truncated) + '</truncated>\n')
            xml.write('\t\t<difficult>' + str(difficult) + '</difficult>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(locs[ind][0]) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(locs[ind][1]) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(locs[ind][0] + locs[ind][2]) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(locs[ind][1] + locs[ind][3]) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')

        xml.write('</annotation>')
    xml.close()
    print("The {} accomplish!".format(filePath))


def autoLabeling(savePath, img, imgName=None, extraName=None, depth=3, truncated=0, difficult=0):
    """
    生成标注图片img的xml文件，
    :param savePath: 保存路径
    :param img: 图片
    :param imgName: 图片的名字
    :param extraName: 附加额外的名字：默认是从000001.png开始命名图片；如果该参数为hsh，则命名从hsh_000001.png开始
    :param depth: 图片的深度
    :param truncated: 是否被截断（0表示完整）
    :param difficult: 目标是否难以识别（0表示容易识别）
    :return:
    """
    global savePathFolderNum
    if savePathFolderNum == -1:
        try:
            savePathFolderNum = len(os.listdir(savePath))
        except FileNotFoundError:
            os.mkdir(savePath)
            savePathFolderNum = 0
    savePathFolderNum = savePathFolderNum + 1
    if imgName == None:
        if extraName != None:
            imgName = extraName + '_' + "{:0>6d}".format(savePathFolderNum)
        else:
            imgName = "{:0>6d}".format(savePathFolderNum)

    imgPath = os.path.join(savePath, imgName + ".png")
    xmlPath = os.path.join(savePath, imgName + ".xml")

    locs, labels = getLabelInfo(img)
    if len(labels) > 0:
        vocXMLFormat(xmlPath, imgName + ".png", img.shape[1], img.shape[0], labels, locs, depth, truncated, difficult)
        cv2.imwrite(imgPath, img)

    return locs, labels


def getLabelInfo(img):
    """
    实现图片的检测，并返回标注信息,eg:
        labels = ["person", "TV"]
        locs = [[24, 32, 156, 145], [124, 76, 472, 384]]
    :param img:
    :return: 返回labels和locs，格式：["目标1","目标2"...]和[[x,y,w,h],[x,y,w,h]...]
    """
    yolov3 = yoloMain.getYolov3()
    boxes, labels, confs, timeLabel = yoloMain.runningYolov3(yolov3, img)
    locs = []
    if boxes == []:
        return locs, labels
    for x1, y1, x2, y2 in boxes:
        w = x2 - x1
        h = y2 - y1
        locs.append([x1, y1, w, h])

    return locs, labels


def showAnnotions(image, locs, labels):
    """
    显示标注
    :param locs:
    :param labels:
    :return:
    """
    for ind, (x, y, w, h) in enumerate(locs):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, labels[ind], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)


def videoAnnotation(videoPath, savePath, gap=10):
    """
    自动标注视频数据
    :param videoPath: 视频的路径
    :param savePath: 标注后保存的路径
    :param gap: 每多少帧才标注
    :return:
    """
    cap = cv2.VideoCapture(videoPath)
    frameNum = 0
    while True:
        ok, img = cap.read()
        frameNum = frameNum + 1
        if not ok:
            break
        if frameNum % gap != 0:
            continue

        locs, labels = autoLabeling(savePath, img)
        showAnnotions(img, locs, labels)

        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()  # 关闭摄像头
            break

    cv2.destroyAllWindows()


def imagesAnnotation(imagesPath, savePath):
    """
    图片自动标注
    :param imagesPath:
    :param savePath:
    :return:
    """
    imagesList = os.listdir(imagesPath)
    for imageName in imagesList:
        imagePath = os.path.join(imagesPath, imageName)
        image = cv2.imread(imagePath)

        locs, labels = autoLabeling(savePath, image, imageName)
        showAnnotions(image, locs, labels)

        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


videPath = "./videos/004.avi"
imagesPath = "C:/Users/weiz/Desktop/images"         # 图片存储文件夹
savePath = "C:/Users/weiz/Desktop/annotions"        # 该文件可以不存在，会自动创建
savePathFolderNum = -1                              # 所存路径文件的个数，-1表示还没有读取
if __name__ == "__main__":
    # videoAnnotation(videPath, savePath, 10)

    imagesAnnotation(imagesPath, savePath)