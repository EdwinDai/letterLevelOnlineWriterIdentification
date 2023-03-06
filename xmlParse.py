import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import os

dataaddr = r"E:\file\Code\Python\datasets\original-xml-part\original"
fileaddr = r'E:\file\Code\Python\datasets\original-xml-part\dataset.txt'
testaddr = r"E:\file\Code\Python\datasets\original-xml-part\original\a01\a01-001\strokesz.xml"
txtaddr = r"E:\file\Code\Python\datasets\original-xml-part\dataset.txt"


# convert xml data address into txt
def convertXml2Txt(dataaddr=dataaddr):
    letters = os.listdir(dataaddr)
    with open(file=fileaddr, mode='a') as f:
        for letter in letters:
            strokepath = os.path.join(dataaddr, letter)
            subletters = os.listdir(strokepath)
            for subletter in subletters:
                letterpath = os.path.join(strokepath, subletter, 'strokesz.xml')
                f.write(letterpath + '\n')


# get stroke address Array
def getTxtaddr():
    addrArray = []
    with open(file=txtaddr, mode='r') as f:
        addrlines = f.readlines()
        for addrline in addrlines:
            addrline = addrline[:-1]
            addrArray.append(addrline)
    return addrArray


# parse xml
def parseXml(addrArray=dataaddr):
    writerArray = []
    for addr in addrArray:
        tree = ET.parse(addr)
        root = tree.getroot()

        # 作者编号加入数组,确保不重复加入，根据数组index获取label
        writerID = root.find('General').find('Form').get('writerID')
        if writerID not in writerArray:
            writerArray.append(writerID)
        Strokes = root.find('StrokeSet').findall('Stroke')

        # 笔画集合
        strokeArr = []
        for idx, stroke in enumerate(Strokes):
            # 点集合50个点
            pointArr = []
            points = stroke.findall('Point')
            # 排除小于50个点的笔画
            if len(points) > 49:
                for idx2, point in enumerate(points):
                    pointArr.extend([point.get('x'), point.get('y')])
                    # 选前50个点
                    if idx2 == 49:
                        break
                strokeArr.append(pointArr)

        arr = np.array(strokeArr)
        print(arr[0])
        print(writerArray.index(writerID))
        break


if __name__ == '__main__':
    parseXml(getTxtaddr())
