import os, torch
import numpy as np
from torch import nn
from torch.optim import lr_scheduler as lrs
from models import SuperSeg

from Util import RestoreNetImg, SoftDiceLoss, BCEFocalLoss

logger = None

class Trainer:
    def __init__(self,
                 data_loader,
                 test_loader,
                 scheduler=lrs.StepLR,
                 dev='cuda:0', devid=0):
        self.dataLoader = data_loader
        self.testLoader = test_loader
        self.dev = dev
        self.cudaid = devid

        # Loss function
        self.focalLoss = BCEFocalLoss(gamma=2,alpha=0.7)
        self.focalLoss = self.focalLoss.cuda(self.cudaid)

        self.precise_dice_loss = SoftDiceLoss()
        self.precise_dice_loss = self.precise_dice_loss.cuda(self.cudaid)

        # Loss weights
        self.lambda_adv = 1
        self.lambda_cycle = 10
        self.lambda_gp = 10

        # Initialize generator and discriminator
        self.super_net = SuperSeg()
        #self.super_net.load_state_dict(torch.load('./saved_models/supernet_700.pth', map_location='cuda:0'))
        self.super_net.cuda(self.cudaid)

        # Optimizers
        self.optimizer_G = torch.optim.Adam([{'params': self.super_net.parameters(),  \
                                              'initial_lr': 0.00002}], lr=0.00002)

        self.scheduler_G = scheduler(self.optimizer_G, step_size=1500, gamma=0.9, last_epoch=-1)

    def Train(self, turn=2):
        self.shot = -1
        torch.set_grad_enabled(True)

        for t in range(turn):
            # if self.gclip > 0:
            #     utils.clip_grad_value_(self.net.parameters(), self.gclip)
            for kk, (lowImg, binImg, meanVal, stdVal) in enumerate(self.dataLoader):
                # torch.cuda.empty_cache()
                self.shot = self.shot + 1
                # torch.cuda.empty_cache()
                self.scheduler_G.step()

                lrImg = lowImg.cuda(self.cudaid)
                binImg = binImg.cuda(self.cudaid)

                if True:
                    self.optimizer_G.zero_grad()

                    # Generate a batch of images
                    seg_A = self.super_net(lrImg)

                    preciseFocalLoss = 50*self.focalLoss(seg_A, binImg)
                    # preciseBCELoss = self.precise_bce_loss(seg_A,binImg)
                    preciseDiceLoss = self.precise_dice_loss(seg_A,binImg)
                    # seg_loss = preciseDiceLoss + preciseBCELoss+preciseFocalLoss
                    seg_loss = preciseDiceLoss + preciseFocalLoss
                    seg_loss.backward()
                    self.optimizer_G.step()

                if self.shot % 10 == 0:
                    lr = self.scheduler_G.get_lr()[0]
                    # print("\r[Epoch %d] [Batch %d] [LR:%f][Focal Loss: %f] [BCE loss: %f] [Dice loss: %f]"
                    print("\r[Epoch %d] [Batch %d] [LR:%f][Focal Loss: %f] [Dice loss: %f]"
                                            % (
                                                t,
                                                self.shot,
                                                lr,
                                                preciseFocalLoss.item(),
                                                # preciseBCELoss.item(),
                                                preciseDiceLoss.item(),
                                            )
                                        )

                    reImg = (seg_A).cpu().data.numpy()[0, 0, :, :, :]
                    reImg2XY = np.max(reImg, axis=0)
                    reImg2XZ = np.max(reImg, axis=1)
                    reImg2XY = (reImg2XY * 254).astype(np.uint8)
                    reImg2XZ = (reImg2XZ * 254).astype(np.uint8)
                    logger.img('segImg2XY', reImg2XY)
                    logger.img('segImg2XZ', reImg2XZ)
                    # interpolate
                    lrImg2 = lowImg.cpu().data.numpy()[0, 0, :, :, :]
                    zoom2 = RestoreNetImg(lrImg2, meanVal.item(), stdVal.item())
                    zoom2XY = np.max(zoom2, axis=0)
                    logger.img('lrImgXY', zoom2XY)
                    zoom2XZ = np.max(zoom2, axis=1)
                    logger.img('lrImgXZ', zoom2XZ)
                    highImgXY = np.max(binImg.cpu().data.numpy()[0, 0, :, :, :], axis=0)
                    highImgXY = RestoreNetImg(highImgXY, 0, 1)
                    logger.img('binImgXY', highImgXY)
                    highImgXZ = np.max(binImg.cpu().data.numpy()[0, 0, :, :, :], axis=1)
                    highImgXZ = RestoreNetImg(highImgXZ, 0, 1)
                    logger.img('binImgXZ', highImgXZ)
                    lossVal = np.float(seg_loss.cpu().data.numpy())
                    if np.abs(lossVal) > 1.5:
                        print('G loss > 1.5')
                    else:
                        logger.plot('G_loss', lossVal)

                if self.shot != 0 and self.shot % 100 == 0:
                    if not os.path.exists('saved_models/'):
                        os.mkdir('saved_models/')
                    torch.save(self.super_net.state_dict(), "saved_models/supernet_%d.pth" % (self.shot))





