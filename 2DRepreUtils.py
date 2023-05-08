import os
import matplotlib.pyplot as plt
import matplotlib
from itertools import product, combinations
import numpy as np
import cv2
import time
import torch.nn as nn
import torch

rootPath = r'Task2'
fileaddr = r'Task2/U29S16.TXT'
picPath = r'Task2Pics'


def parseTxtdata(fileaddr):
    '''
    解析文件
    :param filePath:文件绝对路径
    :return: list [[x,y],[x,y]....]
    '''
    with open(fileaddr) as f:
        # coordinate = []
        xlist = []
        ylist = []
        allist = []
        azlist = []
        flist = []
        txt = f.readlines()
        txt = txt[1:]
        for idx, line in enumerate(txt):
            lineData = line.split(' ')
            xlist.append(float(lineData[0]))
            ylist.append(float(lineData[1]))
            allist.append(float(lineData[4]))
            azlist.append(float(lineData[5]))
            flist.append(float(lineData[6]))

            # pointdata = [float(lineData[0]), float(lineData[1]), float(lineData[3]), float(lineData[4]), float(lineData[5]), float(lineData[6])]
            # coordinate.append(pointdata)
        return xlist, ylist, allist, azlist, flist


def calcMinMax(coordinate):
    xmin = min(coordinate[0])
    xmax = max(coordinate[0])
    ymin = min(coordinate[1])
    ymax = max(coordinate[1])
    almin = min(coordinate[2])
    almax = max(coordinate[2])
    azmin = min(coordinate[3])
    azmax = max(coordinate[3])
    fmin = min(coordinate[4])
    fmax = max(coordinate[4])
    coordinate1 = []
    coordinate2 = []
    coordinate3 = []
    for x in range(len(coordinate[0])):
        coordinate[0][x] = int((coordinate[0][x] - xmin) / (xmax - xmin) * 489)

    for y in range(len(coordinate[1])):
        coordinate[1][y] = int((coordinate[1][y] - ymin) / (ymax - ymin) * 217)

    for al in range(len(coordinate[2])):
        coordinate[2][al] = int((coordinate[2][al] - almin) / (almax - almin) * 255)

    for az in range(len(coordinate[3])):
        coordinate[3][az] = int((coordinate[3][az] - azmin) / (azmax - azmin) * 255)

    for f in range(len(coordinate[4])):
        coordinate[4][f] = int((coordinate[4][f] - fmin) / (fmax - fmin) * 255)

    for i in range(len(coordinate[0])):
        coordinate1.append([coordinate[0][i] + 5, 217 - coordinate[1][i] + 5, coordinate[2][i]])
        coordinate2.append([coordinate[0][i] + 5, 217 - coordinate[1][i] + 5, coordinate[3][i]])
        coordinate3.append([coordinate[0][i] + 5, 217 - coordinate[1][i] + 5, coordinate[4][i]])

    return coordinate1, coordinate2, coordinate3


def cvtImg(coordinate1, coordinate2, coordinate3):
    img = np.zeros((227, 499, 3), dtype=np.uint8)
    for i in range(len(coordinate1)):
        img[coordinate1[i][1]][coordinate1[i][0]] = [coordinate1[i][2], coordinate2[i][2], coordinate3[i][2]]
    # img = img.reshape((1,1,2))
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    return img


if __name__ == '__main__':
    txtNameList = os.listdir(rootPath)
    for txtName in txtNameList:
        coordinate = parseTxtdata(os.path.join(rootPath, txtName))
        a, b, c = calcMinMax(coordinate)
        image = cvtImg(a, b, c)
        fileaddr = os.path.join(rootPath, txtName)
        picName = txtName[:-4]
        cv2.imwrite(os.path.join(picPath, picName + '.jpg'), image)
