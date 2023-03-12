from torch.utils.data import Dataset, DataLoader
import torch
import os
from utils import mixTxtDifferentWriter, parseTxt2data, trim2length, move2TopLeft

rootPath = r'Task1'
# rootPath = r'Task1Para8'


# rootPath = r'E:\file\Code\Python\datasets\Task1\Task1'
# rootPath = '../datasets/Task1'
# fileaddr = r'E:\file\Code\Python\datasets\Task1\Task1\U1S1.TXT'
# fileNameList = ['U1S1.TXT', 'U1S30.TXT', 'U1S20.TXT', 'U14S40.TXT', 'U35S16.TXT', 'U15S3.TXT']


class Dataset_SVC2004(Dataset):
    def __init__(self):
        self.txtList = mixTxtDifferentWriter(40)

    def __getitem__(self, index):
        labelFileName, testFilename, label = self.txtList[index]
        labelPath = os.path.join(rootPath, labelFileName + '.TXT')
        testPath = os.path.join(rootPath, testFilename + '.TXT')

        labelData = trim2length(move2TopLeft(labelPath))
        testData = trim2length(move2TopLeft(testPath))

        labelData = torch.tensor(labelData, dtype=torch.float)
        testData = torch.tensor(testData, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return (labelData, testData), label

    def __len__(self):
        return len(self.txtList)


if __name__ == '__main__':
    dataset = Dataset_SVC2004()
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for x, y in train_dataloader:
        x1, x2 = x
        print(x1.shape)
        print(x2.shape)
        print(y.shape)
        break

    pass
