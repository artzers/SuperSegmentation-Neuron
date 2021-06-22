import os, torch
import numpy as np
from torch import nn
from torch import functional as F
import tifffile
from tqdm import tqdm

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        #pt = torch.sigmoid(_input)
        pt = _input
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1-self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # if self.alpha:
        #     loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):#logits,
        num = targets.size(0)
        smooth = 1

        #probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class DataPackage:
    def __init__(self, lrDir, hrDir, m = 0, s = 0, p = 0.5):
        self.lrDir = lrDir
        self.hrDir = hrDir
        self.meanVal = m
        self.stdVal = s
        self.prob = p

    def SetMean(self, val):
        self.meanVal = val

    def SetStd(self, val):
        self.stdVal = val

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def default_conv3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def prepare(dev, *args):
    # print(dev)
    device = torch.device(dev)
    if dev == 'cpu':
        device = torch.device('cpu')
    return [a.to(device) for a in args]

def RestoreNetImg(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    maxVal = np.max(rImg)
    minVal = np.min(rImg)
    if maxVal <= minVal:
        rImg *= 0
    else:
        rImg = 255./(maxVal - minVal) * (rImg - minVal)
        rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

def RestoreNetImgV2(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

class WDSRBBlock3D(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(WDSRBBlock3D, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv3d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv3d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv3d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        # res = self.body(x) * self.res_scale
        # res += x
        res = self.body(x) + x
        return res

class ResBlock3D(nn.Module):
    def __init__(self,
                 conv=default_conv3d,
                 n_feats=64,
                 kernel_size=3,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(inplace=True),  # nn.LeakyReLU(inplace=True),
                 res_scale=1):

        super(ResBlock3D, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ConvLayer(nn.Module):
    def __init__(self,
                 inplane = 64,
                 n_feats=32,
                 stride = 1,
                 kernel_size=3,
                 bias=True,
                 bn=nn.BatchNorm3d,
                 padding = 1,
                 act=nn.ReLU(inplace=True),  # nn.LeakyReLU(inplace=True),
                 res_scale=1):

        super(ConvLayer, self).__init__()
        m = []
        m.append(nn.Conv3d(inplane, n_feats,kernel_size = kernel_size,
                           stride = stride,padding = padding, bias=bias))
        if bn is not None:
            m.append(bn(n_feats))
        if act is not None:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res

class UpLayer(nn.Module):
    def __init__(self,
                 inplane = 64,
                 n_feats=32,
                 scale_factor=2,
                 bn = nn.BatchNorm3d,
                 act=nn.ReLU(inplace=True)  # nn.LeakyReLU(inplace=True),
                 ):

        super(UpLayer, self).__init__()
        m = []
        m.append(nn.Upsample(scale_factor=scale_factor,mode='trilinear'))

        m.append(nn.Conv3d(in_channels=inplane,out_channels = n_feats,
                           kernel_size=3,padding=3//2 ))
        if bn is not None:
            m.append(bn(n_feats))
        m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res


class PixelUpsampler3D(nn.Module):
    def __init__(self,
                 upscale_factor,
                 # conv=default_conv3d,
                 # n_feats=32,
                 # kernel_size=3,
                 # bias=True
                 ):
        super(PixelUpsampler3D, self).__init__()
        self.scaleFactor = upscale_factor

    def _pixel_shuffle(self, input, upscale_factor):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        channels //= upscale_factor[0] * upscale_factor[1] * upscale_factor[2]
        out_depth = in_depth * upscale_factor[0]
        out_height = in_height * upscale_factor[1]
        out_width = in_width * upscale_factor[2]
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor[0], upscale_factor[1], upscale_factor[2], in_depth,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)

    def forward(self, x):
        # x = self.conv(x)
        up = self._pixel_shuffle(x, self.scaleFactor)
        return up

class GetMemoryDataSetAndCrop:
    def __init__(self, lrDir, hrDir,lineDir, cropSize, epoch):
        self.lrDir = lrDir
        #self.midDir = midDir
        self.hrDir = hrDir
        self.lineDir = lineDir

        self.epoch = epoch

        self.lrFileList = []
        #self.midFileList = []
        self.hrFileList = []
        #self.lineFileList = []

        self.lrImgList = []
        #self.midImgList = []
        self.hrImgList = []
        self.lineImgList = []

        self.beg = [0, 0, 0]
        self.cropSz = cropSize

        for file in os.listdir(self.lrDir):
            if file.endswith('.tif'):
                self.lrFileList.append(file)

        # for file in os.listdir(self.midDir):
        #     if file.endswith('.tif'):
        #         self.midFileList.append(file)

        for file in os.listdir(self.hrDir):
            if file.endswith('.tif'):
                self.hrFileList.append(file)

        self.crossPlace = []
        for file in os.listdir(self.lineDir):
            if file.endswith('.swc'):
                self.crossPlace.append(np.loadtxt(os.path.join(self.lineDir,file)))

        if len(self.lrFileList) != len(self.hrFileList):
            self.check = False

        for name in tqdm(self.lrFileList):
            lrName = os.path.join(self.lrDir,name)
            #midName = os.path.join(self.midDir, name)
            hrName = os.path.join(self.hrDir, name)
            #lineName = os.path.join(self.lineDir, name)
            lrImg = tifffile.imread(lrName)
            #midImg = tifffile.imread(midName)
            hrImg = tifffile.imread(hrName)
            #lineImg = tifffile.imread(lineName)

            lrImg = np.expand_dims(lrImg, axis=0)
            #midImg = np.expand_dims(midImg, axis=0)
            hrImg = np.expand_dims(hrImg, axis=0)
            #lineImg = np.expand_dims(lineImg, axis=0)

            #lrImg = lrImg.astype(np.float)
            #midImg = midImg.astype(np.float)
            #hrImg = hrImg.astype(np.float)

            #midImg = midImg / 255
            #hrImg = hrImg / 255

            self.lrImgList.append(lrImg)
            #self.midImgList.append(midImg)
            self.hrImgList.append(hrImg)
            #self.lineImgList.append(lineImg)

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.hrFileList)

    def __len__(self):
        return self.epoch#len(self.hrFileList)

    def len(self):
        return self.epoch#len(self.hrFileList)

    def __getitem__(self, ind):
        flag = True
        while flag:
            ind = np.random.randint(len(self.hrFileList),len(self.hrFileList)+len(self.crossPlace))
            #ind = np.random.randint(44, len(self.hrFileList))
            if ind < len(self.hrFileList):
                # if ind > 2 and ind < len(self.hrFileList)+8:
                #     ind = 3
                sz = self.lrImgList[ind].shape
                self.beg[0] = np.random.randint(0, sz[1] - self.cropSz[0] - 1)
                self.beg[1] = np.random.randint(0, sz[2] - self.cropSz[1] - 1)
                self.beg[2] = np.random.randint(0, sz[3] - self.cropSz[2] - 1)

            else:
                if ind < len(self.hrFileList)+len(self.crossPlace):
                    curInd = ind-len(self.hrFileList)
                else:
                    curInd = len(self.hrFileList)-1
                ind = curInd
                sz = self.lrImgList[curInd].shape
                placeInd = np.random.randint(0,len(self.crossPlace[curInd]))
                curPlace = (np.round(self.crossPlace[curInd][placeInd])).astype(np.int)
                xMin = np.minimum(np.maximum(0, curPlace[2]-12),sz[3] - self.cropSz[2] - 3)
                xMax = np.maximum(np.minimum(sz[3] - self.cropSz[2] - 1, curPlace[2] - 5),1)
                yMin = np.minimum(np.maximum(0, curPlace[3]-12),sz[2] - self.cropSz[1] - 3)
                yMax = np.maximum(np.minimum(sz[2] - self.cropSz[1] - 1, curPlace[3] - 5),1)
                zMin = np.minimum(np.maximum(0, curPlace[4]-8),sz[1] - self.cropSz[0] - 3)
                zMax = np.maximum(np.minimum(sz[1] - self.cropSz[0] - 1, curPlace[4] - 3),1)
                self.beg[0] = np.random.randint(zMin, zMax)
                self.beg[1] = np.random.randint(yMin, yMax)
                self.beg[2] = np.random.randint(xMin, xMax)

            hrImg = self.hrImgList[ind][:, self.beg[0] * 4:self.beg[0] * 4 + self.cropSz[0] * 4,
                    self.beg[1] * 2:self.beg[1] * 2 + self.cropSz[1] * 2,
                    self.beg[2] * 2:self.beg[2] * 2 + self.cropSz[2] * 2]

            if np.sum(hrImg) < 5100:
                pass
            else:
                lrImg = self.lrImgList[ind][:, self.beg[0]:self.beg[0] + self.cropSz[0],
                        self.beg[1]:self.beg[1] + self.cropSz[1],
                        self.beg[2]:self.beg[2] + self.cropSz[2]]
                flag = False


        rid = np.random.randint(0,6)
        if rid == 0:
            pass#return lrImg, midImg, hrImg
        if rid == 1:
            lrImg,hrImg = lrImg[:,::-1,:,:], hrImg[:,::-1,:,:]
        if rid == 2:
            lrImg,hrImg =  lrImg[:,:,::-1,:], hrImg[:,:,::-1,:]
        if rid == 3:
            lrImg,hrImg =  lrImg[:,:,:,::-1], hrImg[:,:,:,::-1]
        if rid == 4:
            lrImg,hrImg = lrImg[:,::-1,::-1,:],  hrImg[:,::-1,::-1,:]
        if rid == 5:
            lrImg,hrImg =  lrImg[:,:,::-1,::-1], hrImg[:,:,::-1,::-1]

        lrImg = torch.from_numpy(lrImg.copy().astype(np.float)).float()
        hrImg = torch.from_numpy(hrImg.copy().astype(np.float)).float()
        hrImg = hrImg / 255
        return lrImg,  hrImg


class GetMultiTypeMemoryDataSetAndCrop:
    def __init__(self, dataList, cropSize, epoch):
        self.dataList:DataPackage = dataList
        self.lrImgList = [[] for x in range(len(self.dataList))]
        self.hrImgList = [[] for x in range(len(self.dataList))]

        self.randProbInteval = [0 for x in range(len(self.dataList) + 1)]
        for k in range(1,len(self.dataList)+1):
            self.randProbInteval[k] = self.dataList[k-1].prob * 100 + self.randProbInteval[k-1]

        self.epoch = epoch

        self.beg = [0, 0, 0]
        self.cropSz = cropSize

        for k in range(len(self.dataList)):
            pack = self.dataList[k]
            lrDir = pack.lrDir
            hrDir = pack.hrDir
            lrFileList = []
            hrFileList = []

            for file in os.listdir(lrDir):
                if file.endswith('.tif'):
                    lrFileList.append(file)

            for file in os.listdir(hrDir):
                if file.endswith('.tif'):
                    hrFileList.append(file)

            for name in tqdm(lrFileList):
                lrName = os.path.join(lrDir,name)
                hrName = os.path.join(hrDir, name)
                lrImg = tifffile.imread(lrName)
                hrImg = tifffile.imread(hrName)

                lrImg = np.expand_dims(lrImg, axis=0)
                hrImg = np.expand_dims(hrImg, axis=0)

                self.lrImgList[k].append(lrImg)
                self.hrImgList[k].append(hrImg)

    def __len__(self):
        return self.epoch#len(self.hrFileList)

    def len(self):
        return self.epoch#len(self.hrFileList)

    def __getitem__(self, ind):
        flag = True
        dataID = 0
        randNum = np.random.randint(self.randProbInteval[-1])#len(self.dataList)
        for k in range(len(self.randProbInteval)-1):
            if self.randProbInteval[k] < randNum < self.randProbInteval[k + 1]:
                dataID = k
                break

        ind = np.random.randint(len(self.lrImgList[dataID]))
        tryNum = 0
        while flag:
            sz = self.lrImgList[dataID][ind].shape
            self.beg[0] = np.random.randint(0, sz[1] - self.cropSz[0] - 1)
            self.beg[1] = np.random.randint(0, sz[2] - self.cropSz[1] - 1)
            self.beg[2] = np.random.randint(0, sz[3] - self.cropSz[2] - 1)

            hrImg = self.hrImgList[dataID][ind][:, self.beg[0] * 4:self.beg[0] * 4 + self.cropSz[0] * 4,
                    self.beg[1] * 2:self.beg[1] * 2 + self.cropSz[1] * 2,
                    self.beg[2] * 2:self.beg[2] * 2 + self.cropSz[2] * 2]

            if np.sum(hrImg) < 800 and tryNum < 20:
                tryNum += 1
            else:
                lrImg = self.lrImgList[dataID][ind][:, self.beg[0]:self.beg[0] + self.cropSz[0],
                        self.beg[1]:self.beg[1] + self.cropSz[1],
                        self.beg[2]:self.beg[2] + self.cropSz[2]]
                flag = False


        rid = np.random.randint(0,6)
        if rid == 0:
            pass#return lrImg, midImg, hrImg
        if rid == 1:
            lrImg,hrImg = lrImg[:,::-1,:,:], hrImg[:,::-1,:,:]
        if rid == 2:
            lrImg,hrImg =  lrImg[:,:,::-1,:], hrImg[:,:,::-1,:]
        if rid == 3:
            lrImg,hrImg =  lrImg[:,:,:,::-1], hrImg[:,:,:,::-1]
        if rid == 4:
            lrImg,hrImg = lrImg[:,::-1,::-1,:],  hrImg[:,::-1,::-1,:]
        if rid == 5:
            lrImg,hrImg =  lrImg[:,:,::-1,::-1], hrImg[:,:,::-1,::-1]

        lrImg = torch.from_numpy(lrImg.copy().astype(np.float)).float()
        hrImg = torch.from_numpy(hrImg.copy().astype(np.float)).float()
        lrImg = (lrImg - self.dataList[dataID].meanVal) / self.dataList[dataID].stdVal
        hrImg = hrImg / 255.
        return lrImg,  hrImg , self.dataList[dataID].meanVal, self.dataList[dataID].stdVal
