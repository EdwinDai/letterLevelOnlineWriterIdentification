import os
import matplotlib.pyplot as plt
import matplotlib
from itertools import product

# matplotlib.use('TkAgg')
matplotlib.rc("font", family='Microsoft YaHei')

# rootPath = r'Task1'

# rootPath = r'E:\file\Code\Python\datasets\Task1\Task1'


# fileaddr = r'E:\file\Code\Python\datasets\Task1\Task1\U1S1.TXT'
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
        txt = txt[1:]
        for idx, line in enumerate(txt):
            lineData = line.split(' ')
            coordinate.append([int(lineData[0]), int(lineData[1]), int(lineData[3])])
        while (len(coordinate) < 300):
            coordinate.append([0, 0, 0])
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


# if __name__ == '__main__':
#     res = mixTxtDifferentWriter(40)
#     print(len(res))
