
import os, tifffile
import torch
import numpy as np
from models import SuperSeg
import time

root = './data/'
dirList = os.listdir(root)
# dirList=['00000_00000_00236']

lowMeanVal = 35.#35.
lowStdVal = 50.


minLowRange=[0,0,0]#[1800,3300,550]#z,y,x
minLowRange = minLowRange[-1::-1]
maxLowRange=[128,128,128]#[120,120,120]
maxLowRange = maxLowRange[-1::-1]#reverse
readRange = [32,32,32]

zMinLowList = []
zMaxLowList = []
yMinLowList = []
yMaxLowList = []
xMinLowList = []
xMaxLowList = []

# for k in range(minLowRange[0], maxLowRange[0] - 15, 16):
#     zMinLowList.append(k)
#     zMaxLowList.append(k+16)
#
# for k in range(minLowRange[1], maxLowRange[1] - 31,16):
#     yMinLowList.append(k)
#     yMaxLowList.append(k+32)#59
#
# for k in range(minLowRange[2], maxLowRange[2] - 31, 16):
#     xMinLowList.append(k)
#     xMaxLowList.append(k+32)

for k in range(minLowRange[0], maxLowRange[0] - (readRange[0] - 1), readRange[0]): #TODO:
    zMinLowList.append(k)
    zMaxLowList.append(k+readRange[0])

for k in range(minLowRange[1], maxLowRange[1] - (readRange[1] - 1), readRange[1]):
    yMinLowList.append(k)
    yMaxLowList.append(k+readRange[1])#59

for k in range(minLowRange[2], maxLowRange[2] - (readRange[2] - 1), readRange[2]):
    xMinLowList.append(k)
    xMaxLowList.append(k+readRange[2])

pretrained_net = SuperSeg()
pretrained_net.load_state_dict(torch.load('./saved_models/ZHM_19500.pth',map_location='cuda:0'))
pretrained_net = pretrained_net.cuda(0)
pretrained_net.eval()
torch.set_grad_enabled(False)
torch.cuda.empty_cache()

for dirName in dirList:
    readPath = os.path.join(root,'%s.tif'%dirName)
    savePath = os.path.join(root,'%s_zhou.tif'%dirName)

    img = tifffile.imread(readPath)

    xBase = xMinLowList[0]
    yBase = yMinLowList[0]
    zBase = zMinLowList[0]
    resImg = np.zeros((img.shape[0]*4, img.shape[1]*2, img.shape[2]*2), dtype=np.float)

    beg = time.time()
    for i in range(len(zMinLowList)):#TODO
        for j in range(len(yMinLowList)):
            for k in range(len(xMinLowList)):
                print('processing %s:%d-%d, %d-%d %d-%d'%(dirName, xMinLowList[k], xMaxLowList[k],
                                                       yMinLowList[j], yMaxLowList[j],
                                                       zMinLowList[i], zMaxLowList[i]))
                lowImg = img[zMinLowList[i]:zMaxLowList[i],
                                          yMinLowList[j]:yMaxLowList[j],
                                          xMinLowList[k]:xMaxLowList[k]]
                #print('Cache : %d'%len(reader.imageCacheManager.cacheList))

                lowImg = np.array(lowImg, dtype=np.float32)
                lowImg = (lowImg - lowMeanVal) / (lowStdVal)
                lowImg = np.expand_dims(lowImg, axis=0)
                lowImg = np.expand_dims(lowImg, axis=0)
                lowImg = torch.from_numpy(lowImg).float()
                lowImg = lowImg.cuda(0)
                pre2 = pretrained_net(lowImg)
                saveImg = pre2.cpu().data.numpy()[0, 0, :, :, :]
                resImg[zMinLowList[i]*4:zMaxLowList[i]*4,
                            yMinLowList[j]*2:yMaxLowList[j]*2,
                            xMinLowList[k]*2:xMaxLowList[k]*2]\
                    = np.maximum(saveImg, resImg[zMinLowList[i]*4:zMaxLowList[i]*4,
                            yMinLowList[j]*2:yMaxLowList[j]*2,
                            xMinLowList[k]*2:xMaxLowList[k]*2])

    print('time: ',time.time() - beg)

    diff = np.max(resImg)-np.min(resImg)
    resImg -= np.min(resImg)
    resImg /= diff
    resImg *= 254
    print('save as %s'%savePath)
    tifffile.imwrite(savePath, np.uint8(resImg)) #, compress = 6


