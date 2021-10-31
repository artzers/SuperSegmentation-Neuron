import os, tifffile
import numpy as np
from scanf import scanf
from tqdm import tqdm
from scipy.ndimage import zoom
from Predict_10Data import HuangPredict
import shutil

srcPath = 'D:/Document/SuperSeg/fig/'
root = srcPath
dirList = os.listdir('D:/Document/SuperSeg/fig-swc/')

for name in tqdm(dirList):
    dirPath = os.path.join(root, name)
    fileList = os.listdir(dirPath)
    # if os.path.exists(os.path.join(dirPath,name[:-4]+'_allSwc.swc')):
    #     os.remove(os.path.join(dirPath,name[:-4]+'_allSwc.swc'))
    allCurve = []
    tifname = name
    for file in fileList:
        if file[-3:] != 'swc' or file[-10:] == 'shapes.swc':
            continue
        if file[-10:] == 'allSwc.swc':
            continue
        with open(os.path.join(dirPath,file),'r') as fp:
            lines = fp.readlines()
            for line in lines:
                data = scanf('%d %d %f %f %f %f %d\n', line)
                if data[6] != data[0] - 1:
                    allCurve.append([])
                allCurve[-1].append(np.array([data[2],data[3],data[4]]))

    saveName = os.path.join(dirPath,tifname+'_allSwc.swc')
    with open(saveName,'w') as fp:
        id = 1
        for k in range(len(allCurve)):
            fp.write('%d 1 %f %f %f 1 -1\n' % (id,
                                               allCurve[k][0][0],
                                               allCurve[k][0][1],
                                               allCurve[k][0][2]))
            id += 1
            for kk in range(1,len(allCurve[k])):
                fp.write('%d 1 %f %f %f 1 %d\n'%(id,
                                                 allCurve[k][kk][0],
                                                 allCurve[k][kk][1],
                                                 allCurve[k][kk][2], id - 1))
                id += 1
