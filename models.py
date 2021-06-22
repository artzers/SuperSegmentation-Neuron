import torch.nn as nn
import torch.nn.functional as F
import torch
from Util import WDSRBBlock3D, PixelUpsampler3D, ResBlock3D, ConvLayer, UpLayer
import numpy as np

class SuperSeg(nn.Module):
    def __init__(self):
        super(SuperSeg, self).__init__()

        n_resblocks = 4
        kernel_size = 3

        self.act = nn.LeakyReLU()
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.Conv0 = ConvLayer(inplane=1, n_feats=128)

        wdsr = []
        for i in range(n_resblocks):
            wdsr.append(
                WDSRBBlock3D(n_feats=128, kernel_size=kernel_size, wn=wn, act=self.act))  # expand = 3
        wdsr.append(nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3,
                              padding=3 // 2))

        n_feats = 128
        out_feats = 2 * 2 * 4 * 128
        tail = []
        tail.append(
            wn(nn.Conv3d(n_feats, out_feats, 3, padding=3 // 2)))
        tail.append(PixelUpsampler3D((4, 2, 2)))

        self.Up = UpLayer(inplane=1, n_feats=128, scale_factor=(4, 2, 2))

        self.SimpleUp = nn.Upsample(scale_factor=(4, 2, 2), mode='trilinear')

        self.Conv1 = ConvLayer(inplane=1, n_feats=128)
        self.Conv2 = ConvLayer(inplane=257, n_feats=128)
        self.Res1 = ResBlock3D(n_feats=128, bn=True, act=self.act)
        self.Res2 = ResBlock3D(n_feats=128, bn=True, act=self.act)
        self.Res3 = ResBlock3D(n_feats=128, bn=True, act=self.act)
        self.Res4 = ResBlock3D(n_feats=128, bn=True, act=self.act)
        self.Res5 = ResBlock3D(n_feats=128, bn=True, act=self.act)
        self.Res6 = ResBlock3D(n_feats=128, bn=True, act=self.act)

        self.Conv3 = nn.Conv3d(128, 1, kernel_size=3, padding=1)

        self.wdsr = nn.Sequential(*wdsr)
        self.tail = nn.Sequential(*tail)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input):
        scaleOrig = self.SimpleUp(input)

        scale = self.Conv1(scaleOrig)
        scale2 = self.Res1(scale)
        scale2 = self.Res2(scale2)
        scale = scale2 + scale
        x1 = self.Conv0(input)
        x = self.wdsr(x1)
        #x = x+x1
        x = self.tail(x)
        scaleConvOrig = self.Up(input)
        x = x + scaleConvOrig
        x = torch.cat((x, scale, scaleOrig), dim=1)
        x = self.Conv2(x)
        res = self.Res3(x)
        res = self.Res4(res)
        res = res + x
        res1 = self.Res5(res)
        res1 = self.Res6(res1)
        res = res + res1
        res = self.Conv3(res)
        res = torch.sigmoid(res)
        res = torch.clamp(res, min=0.01, max=0.99)
        return res


