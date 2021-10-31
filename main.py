import os, torch
import numpy as np
from vis import vis_tool
import tifffile
from libtiff import TIFFimage
from torch.utils.data import DataLoader
import math,time
from tqdm import tqdm
from Util import GetMultiTypeMemoryDataSetAndCrop, DataPackage
import Net

'''
python -m visdom.server
http://localhost:8097/
'''

def CalcMeanStd(path):
    srcPath = path
    fileList = os.listdir(srcPath)
    fileNum = len(fileList)

    globalMean = 0
    globalStd = 0

    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        mean = np.mean(img)
        # print(mean)
        globalMean += mean
    globalMean /= fileNum


    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        img = img.astype(np.float)
        img -= globalMean
        sz = img.shape[0] * img.shape[1] * img.shape[2]
        globalStd += np.sum(img ** 2) / np.float(sz)
    globalStd = (globalStd / len(fileList)) ** (0.5)

    print(globalMean)
    print(globalStd)
    return globalMean,globalStd


def CalcMeanMax(path):
    srcPath = path
    fileList = os.listdir(srcPath)
    fileNum = len(fileList)

    maxVal = 0
    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        maxVal = np.maximum(maxVal, np.max(img))

    print(maxVal)

    globalMean = 0
    globalStd = 0

    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        mean = np.mean(img)
        globalMean += mean
    globalMean /= fileNum
    print(globalMean)
    return globalMean,maxVal

env = 'SuperSeg'
globalDev = 'cuda:0'
globalDeviceID = 0

if __name__ == '__main__':
    lowMean,lowStd = CalcMeanStd("D:\Document\SuperSeg/fig4\zhm_sample/new\orig/")
    print(lowMean, lowStd)
    # exit(0)

    dataList = []
    dataList.append( DataPackage('D:\Document/fig/orig/',
                        'D:\Document/fig/bin',lowMean,lowStd, 1.0) )

    train_dataset = GetMultiTypeMemoryDataSetAndCrop(dataList, [16, 24, 24], 500)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=0)

    visdomable = True
    if visdomable == True:
        logger = vis_tool.Visualizer(env=env)
        logger.reinit(env=env)

    Net.logger = logger

    trainer = Net.Trainer(data_loader=train_loader, test_loader=None)

    time_start = time.time()
    trainer.Train(turn=50)
    time_end = time.time()
    print('totally time cost', time_end - time_start)

