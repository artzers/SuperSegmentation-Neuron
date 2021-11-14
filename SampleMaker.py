import os, tifffile
import numpy as np
from scanf import scanf
from tqdm import tqdm
from scipy.ndimage import zoom
import shutil

srcPath = 'D:\Document\SuperSeg/fig/'
root = srcPath
dirList = os.listdir(srcPath)

for name in tqdm(dirList):
    curDir = os.path.join(srcPath, name)
    tifFile = os.path.join(curDir, '%s.tif'%name)
    origImg = tifffile.imread(tifFile)
    img = zoom(origImg,(4,2,2))
    swcFile = os.path.join(curDir, '%s_allSwc.swc' % name)
    allCurve = []
    with open(swcFile, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            data = scanf('%d %d %f %f %f %f %d\n', line)
            if data[6] != data[0] - 1:
                allCurve.append([])
            allCurve[-1].append(np.array([data[2]*2, data[3]*2, data[4]*4 +1]))#modified
            #allCurve[-1].append(np.array([data[2], data[3], data[4]]))  # modified
    # intepolate
    newAllCurve = []
    for curve in allCurve:
        sz = len(curve)
        newCurve = []
        for i in range(0, sz - 1):
            newCurve.append(curve[i])
            dis = np.linalg.norm(curve[i] - curve[i + 1])
            idis = int(round(dis)) * 2
            for j in range(1, idis):
                newCurve.append((curve[i + 1] - curve[i]) * float(j) / float(idis) + curve[i])
        newCurve.append(curve[sz - 1])
        newAllCurve.append(newCurve)
    # fill the bin image
    radius = 1
    imgShape = np.array(img.shape)#img has been scaled
    #imgShape[0] = imgShape[0] * 3
    binImg = np.zeros(imgShape, dtype=np.uint8)
    for curve in newAllCurve:
        for node in curve:
            pt = np.round(node)
            pt = pt.astype(np.int)
            pt[0] = np.minimum(pt[0],img.shape[2]-1)
            pt[1] = np.minimum(pt[1], img.shape[1]-1)
            pt[2] = np.minimum(pt[2], img.shape[0]-1)
            curVal = img[pt[2], pt[1], pt[0]]
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    for k in range(-radius, radius + 1):
                        tmp = pt + np.array([i, j, k])
                        tmp[0] = min(tmp[0], imgShape[2] - 1)
                        tmp[0] = max(tmp[0], 0)
                        tmp[1] = min(tmp[1], imgShape[1] - 1)
                        tmp[1] = max(tmp[1], 0)
                        tmp[2] = min(tmp[2], imgShape[0] - 1)
                        tmp[2] = max(tmp[2], 0)
                        dist = np.linalg.norm(node - tmp.astype(np.float))
                        if dist > 1.2:
                            continue
                        tmpVal = img[tmp[2], tmp[1], tmp[0]]
                        if tmpVal / curVal < 0.85:
                            continue
                        binImg[tmp[2], tmp[1], tmp[0]] = 255
    # save image
    saveFile = os.path.join(curDir, '%s_bin.tif' % name)
    tifffile.imwrite(saveFile, binImg)
