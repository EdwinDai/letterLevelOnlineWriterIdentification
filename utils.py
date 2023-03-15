import os
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
import numpy as np
import cv2
import time
import torch.nn as nn
import torch


# 统计时打开
# matplotlib.use('TkAgg')
# matplotlib.rc("font", family='Microsoft YaHei')

# rootPath = r'Task1'

# rootPath = r'E:\file\Code\Python\datasets\Task1\Task1'
# picAddr = r'E:\file\Code\Python\datasets\Task1\pictures'

# fileaddr = r'Task1Para8/U1S1.TXT'


# fileNameList = ['U1S1.TXT', 'U1S30.TXT', 'U1S20.TXT', 'U14S40.TXT', 'U35S16.TXT', 'U15S3.TXT']
# fileName = 'U15S3.TXT'
# destaddr = r'E:\file\Code\Python\datasets\Task1'


def isGenuineOrForgery(fileName):
    '''
    根据S序号判定真假样本
    :param 文件名
    :return 1真 0假
    '''
    sIndex = fileName.index('S')
    SNumber = int(fileName[sIndex + 1:-4])
    if SNumber > 20:
        return 0
    else:
        return 1


def parseTxt2data(filePath):
    '''
    解析文件
    :param filePath:文件绝对路径
    :return: list [[x,y],[x,y]....]
    '''
    with open(filePath) as f:
        coordinate = []
        txt = f.readlines()
        # txt = txt[1:]
        for idx, line in enumerate(txt):
            lineData = line.split(' ')
            # 3维数据
            # coordinate.append([float(lineData[0]), float(lineData[1]), float(lineData[3])])
            # 8维数据
            coordinate.append(
                [float(lineData[0]), float(lineData[1]), float(lineData[2]), float(lineData[3]), float(lineData[4]),
                 float(lineData[5]), float(lineData[6]), float(lineData[7][:-1])])
        # 补0/截断至300长度
        # coordinate = trim2length(coordinate)
        return coordinate


def trim2length(coordinate):
    while (len(coordinate) < 300):
        # coordinate.append([0, 0, 0])
        coordinate.append([0, 0, 0, 0, 0, 0, 0, 0])

    if len(coordinate) > 300:
        coordinate = coordinate[:300]
    return coordinate


def calcSeqLength(rootPath):
    '''
        统计数据集中各签名序列长度
        :param rootPath:str
        :return list:int
        '''
    txtNameList = os.listdir(rootPath)
    countList = []
    for txtName in txtNameList:
        txtPath = os.path.join(rootPath, txtName)
        with open(txtPath) as f:
            length = int(f.readline())
            countList.append(length)
    return countList


def getLongestSeq(rootPath):
    '''
    返回数据集中最长序列长度
    :param rootPath
    :return int 793个点
    '''
    txtNameList = os.listdir(rootPath)
    max = 0
    filename = ''
    for txtName in txtNameList:
        txtPath = os.path.join(rootPath, txtName)
        with open(txtPath) as f:
            length = int(f.readline())
            if length > max:
                max = length
                filename = txtName
    return max, filename


def calcStrokes(rootPath):
    '''
    统计数据集中各签名笔画数
    :param rootPath:str
    :return list:int
    '''
    txtNameList = os.listdir(rootPath)
    countList = []
    for txtName in txtNameList:
        txtPath = os.path.join(rootPath, txtName)
        singleTxtCount = 0
        with open(txtPath) as f:
            lines = f.readlines()[1:]
            for line in lines:
                lineData = int(line.split(' ')[3])
                if lineData == 0:
                    singleTxtCount += 1
        countList.append(singleTxtCount)
    return countList


def calcStrokeLength(rootPath):
    '''
    统计数据集中签名笔画长度
    :param rootPath:str
    :return list:int
    '''
    txtNameList = os.listdir(rootPath)
    lengthList = []
    for txtName in txtNameList:
        txtPath = os.path.join(rootPath, txtName)
        singleTxtLength = 0
        with open(txtPath) as f:
            lines = f.readlines()[1:]
            for line in lines:
                lineData = int(line.split(' ')[3])
                if lineData == 1:
                    singleTxtLength += 1
                elif lineData == 0 and singleTxtLength != 0:
                    lengthList.append(singleTxtLength)
                    singleTxtLength = 0
            lengthList.append(singleTxtLength)
    return lengthList


def showStatistics(rootPath: str, method: str):
    '''
    展示统计信息图表
    :param rootPath:str
    :param method:str
    :return None
    '''
    # plt.rc("font", family='Microsoft YaHei')
    if method == 'calcStrokes':
        data = calcStrokes(rootPath)
        x = '笔画数量'
        y = '签名数量'
        d = 1
    elif method == 'calcStrokeLength':
        data = calcStrokeLength(rootPath)
        x = '笔画长度'
        y = '笔画数量'
        d = 20
    elif method == 'calcSeqLength':
        data = calcSeqLength(rootPath)
        x = '序列长度'
        y = '序列数量'
        d = 30
    elif method == 'calcSigSize':
        data = calcSeqLength(rootPath)
        x = '签名尺寸'
        y = '签名数量'
        d = 30
    else:
        print('wrong method')
        return

    plt.hist(data, (max(data) - min(data)) // d)
    plt.xticks(range(min(data), max(data) + d, d))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()
    plt.show()


def mixTxtSameWriter(usernum: int):
    '''
    对签名进行两两配对(同一作者名下genuine and skilled forgery)
    :param UserId:
    :return: list[[Sig1,Sig1],[Sig1,Sig2]...]
    '''
    resList = []
    for userId in range(1, usernum + 1):
        labelSigList = []
        testSigList = []
        for i in range(1, 21):
            labelSigList.append('U' + str(userId) + 'S' + str(i))
        for j in range(21, 41):
            testSigList.append('U' + str(userId) + 'S' + str(j))
        trueRes = product(labelSigList, labelSigList)
        falseRes = product(labelSigList, testSigList)
        for x, y in list(trueRes):
            resList.append([x, y, 1])
        for x, y in list(falseRes):
            resList.append([x, y, 0])
    return resList


def mixTxtDifferentWriter(usernum: int):
    '''
    对签名进行两两配对(不同作者名下genuine and forgery)
    :param UserId:
    :return: list[[Sig1,Sig1],[Sig1,Sig2]...]
    '''
    resList = []
    for userId in range(1, usernum + 1):
        labelSigList = []
        testSigList = []
        for i in range(1, 21):
            labelSigList.append('U' + str(userId) + 'S' + str(i))
        for testUserId in range(1, 41):
            if testUserId == userId:
                continue
            if len(testSigList) == 20:
                break
            testSigList.append('U' + str(testUserId) + 'S' + str(1))
        trueRes = product(labelSigList, labelSigList)
        falseRes = product(labelSigList, testSigList)
        for x, y in list(trueRes):
            resList.append([x, y, 1])
        for x, y in list(falseRes):
            resList.append([x, y, 0])
    return resList


def move2TopLeft(fileaddr):
    '''
    将签名移动到左上角
    :param rootPath:
    :return:
    '''
    xmin = 9999
    ymin = 9999
    xmax = 0
    ymax = 0
    signature = parseTxt2data(fileaddr)
    for point_data in signature:
        xmin = point_data[0] if point_data[0] < xmin else xmin
        ymin = point_data[1] if point_data[1] < ymin else ymin

        # xmax = point_data[0] if point_data[0] > xmax else xmax
        # ymax = point_data[1] if point_data[1] > ymax else ymax
    for point_data in signature:
        point_data[0] = int(point_data[0]) - xmin
        point_data[1] = int(point_data[1]) - ymin
    # length = xmax - xmin
    # height = ymax - ymin
    # print(length, height)
    # print(length * height)
    return signature


def calcSigSize(rootPath):
    # 240
    txtNameList = os.listdir(rootPath)
    size_list = []
    for txtName in txtNameList:
        txtPath = os.path.join(rootPath, txtName)
        signature = move2TopLeft(txtPath)
        xmin = 9999
        ymin = 9999
        xmax = 0
        ymax = 0
        length = 0
        height = 0
        for point_data in signature:
            xmin = point_data[0] if point_data[0] < xmin else xmin
            ymin = point_data[1] if point_data[1] < ymin else ymin
            xmax = point_data[0] if point_data[0] > xmax else xmax
            ymax = point_data[1] if point_data[1] > ymax else ymax
            length = xmax - xmin
            height = ymax - ymin
        size_list.append([(length * height) / 100000])
    return size_list


def normalizeSig1(signature):
    '''归一化到[0,1]
    :param:signature
    :return:NormalizedSignature
    '''
    xmax = 0
    ymax = 0
    for point_data in signature:
        xmax = point_data[0] if point_data[0] > xmax else xmax
        ymax = point_data[1] if point_data[1] > ymax else ymax
    for point_data in signature:
        point_data[0] = int(point_data[0]) / xmax
        point_data[1] = int(point_data[1]) / ymax
    return signature


def normalizeSig2(signature):
    '''均值方差归一化
    :param:signature
    :return:NormalizedSignature
    '''
    x_sum1 = 0
    x_sum2 = 0
    y_sum1 = 0
    y_sum2 = 0
    sig_length = len(signature)
    for point_data in signature:
        x_sum1 += point_data[0]
        y_sum1 += point_data[1]
    x_average = x_sum1 / sig_length
    y_average = y_sum1 / sig_length
    for point_data in signature:
        x_sum2 += (point_data[0] - x_average) ** 2
        y_sum2 += (point_data[1] - y_average) ** 2
    x_variance = x_sum2 / sig_length
    y_variance = y_sum2 / sig_length
    for point_data in signature:
        point_data[0] = (point_data[0] - x_average) / (x_variance ** 0.5)
        point_data[1] = (point_data[1] - y_average) / (y_variance ** 0.5)
    return signature


def drawSig(fileaddr, picName):
    image = np.zeros((9999, 9999, 1), np.uint8)  # 创建一个黑色面板
    sig = move2TopLeft(fileaddr)
    for i in range(len(sig) - 1):
        if sig[i][3] != 0:
            cv2.line(image, (sig[i][0], 9999 - sig[i][1]), (sig[i + 1][0], 9999 - sig[i + 1][1]), (255, 0, 0), 3)  # 画直线
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 色彩空间转换
    cv2.imwrite(os.path.join(picAddr, picName + '.jpg'), image)


def get_parameters(sig, i):
    x1 = sig[i][0]
    y1 = sig[i][1]
    x2 = sig[i + 1][0]
    y2 = sig[i + 1][1]
    p2_pen_down = sig[i + 1][3]
    time1 = sig[i][2]
    time2 = sig[i + 1][2]
    time_interval = time2 - time1
    # time_stamp = sig[0][2] - 1

    tangent = (y2 - y1) / ((x2 - x1) if (x2 - x1) != 0 else 1)  # 4
    abs_position = ((x1 ** 2) + (y1 ** 2)) ** 0.5  # 5
    x_velocity = (((x2 - x1) ** 2) ** 0.5) / time_interval  # 6
    y_velocity = (((y2 - y1) ** 2) ** 0.5) / time_interval  # 7
    path_velocity = ((((y2 - y1) ** 2) + ((x2 - x1) ** 2)) ** 0.5) / time_interval  # 8
    point_data = [x1, y1, sig[i][3], tangent, abs_position, x_velocity, y_velocity, path_velocity]
    return point_data


if __name__ == '__main__':
    mixTxtSameWriter(1)
    # a = torch.randn(4, 1, 128)
    # b = torch.randn(64, 32)
    # conv1 = nn.Conv1d(1, 16, 3, padding=1)
    # conv2 = nn.Conv1d(16, 16, 3, padding=1)
    # maxpool = nn.MaxPool1d(2)
    # c = conv1(a)
    # c = maxpool(c)
    # print(c.shape)
    # c = conv2(c)
    # print(c.shape)

    # 生成8params签名

    # newPath = r'E:\file\Code\Python\datasets\Task1\Task1Para8'
    # txtNameList = os.listdir(rootPath)
    # for txtName in txtNameList:
    #     txt_path = os.path.join(rootPath, txtName)
    #     sig = move2TopLeft(fileaddr)
    #     sig = normalizeSig1(sig)
    #     with open(os.path.join(newPath, txtName), 'a') as f:
    #         for i in range(len(sig) - 1):
    #             sig[i] = get_parameters(sig, i)
    #             my_string = ' '.join(map(str, sig[i]))
    #             f.write(my_string + '\n')

    # sig = parseTxt2data(fileaddr)
    # print(sig)
# showStatistics(rootPath, 'calcSigSize')
# txtNameList = os.listdir(rootPath).0
# for txtName in txtNameList:
#     fileaddr = os.path.join(rootPath, txtName)
#     picName = txtName[:-4]
#     drawSig(fileaddr, picName)

#     break
# picName = 'U1S1'
# print(os.path.join(picAddr, picName+'.jpg'))
