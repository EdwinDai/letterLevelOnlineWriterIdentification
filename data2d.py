from torch.utils.data import Dataset, DataLoader
import torch
import os
from utils import trim2length, move2TopLeft, mixTxtSameWriter, mixJpgAABWriter
import cv2

rootPath = r'Task2Pics'


# rootPath = r'Task1Para8'


# rootPath = r'E:\file\Code\Python\datasets\Task1\Task1'
# rootPath = '../datasets/Task1'
# fileaddr = r'E:\file\Code\Python\datasets\Task1\Task1\U1S1.TXT'
# fileNameList = ['U1S1.TXT', 'U1S30.TXT', 'U1S20.TXT', 'U14S40.TXT', 'U35S16.TXT', 'U15S3.TXT']
# icdartraintruepath = r'ICDAR2011/train/Online Genuine'
# icdartrainfalsepath = r'ICDAR2011/train/Online Forgeries'


class Dataset_SVC2004_train(Dataset):
    def __init__(self):
        self.traintxtList = mixJpgAABWriter(1, 33)
        # self.txtList = readicdar(icdartraintruepath, icdartrainfalsepath)

    def __getitem__(self, index):
        anchorFileName, trueFilename, testFilename, label = self.traintxtList[index]
        # labelPath = os.path.join(rootPath, labelFileName)

        # if len(testFilename) > 10:
        #     testPath = os.path.join(icdartrainfalsepath, testFilename)
        # else:
        #     testPath = os.path.join(icdartraintruepath, testFilename)

        anchorFileNamePath = os.path.join(rootPath, anchorFileName + '.jpg')
        truePath = os.path.join(rootPath, trueFilename + '.jpg')
        testPath = os.path.join(rootPath, testFilename + '.jpg')

        anchorData = cv2.imread(anchorFileNamePath)
        trueData = cv2.imread(truePath)
        testData = cv2.imread(testPath)
        # labelData = torch.tensor(labelData, dtype=torch.float)
        # testData = torch.tensor(testData, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return (anchorData, trueData, testData), label

    def __len__(self):
        return len(self.traintxtList)


class Dataset_SVC2004_test(Dataset):
    def __init__(self):
        self.traintxtList = mixJpgAABWriter(34, 37)
        # self.txtList = readicdar(icdartraintruepath, icdartrainfalsepath)

    def __getitem__(self, index):
        anchorFileName, trueFilename, testFilename, label = self.traintxtList[index]
        # labelPath = os.path.join(rootPath, labelFileName)

        # if len(testFilename) > 10:
        #     testPath = os.path.join(icdartrainfalsepath, testFilename)
        # else:
        #     testPath = os.path.join(icdartraintruepath, testFilename)

        anchorFileNamePath = os.path.join(rootPath, anchorFileName + '.jpg')
        truePath = os.path.join(rootPath, trueFilename + '.jpg')
        testPath = os.path.join(rootPath, testFilename + '.jpg')

        anchorData = cv2.imread(anchorFileNamePath)
        trueData = cv2.imread(truePath)
        testData = cv2.imread(testPath)
        # labelData = torch.tensor(labelData, dtype=torch.float)
        # testData = torch.tensor(testData, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return (anchorData, trueData, testData), label

    def __len__(self):
        return len(self.traintxtList)


if __name__ == '__main__':
    dataset = Dataset_SVC2004_test()
    print(len(dataset))
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for x, y in train_dataloader:
        x1, x2, x3 = x
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(y.shape)
        break