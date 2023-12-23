from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
from clustercontrast.utils.board_writter import BoardWriter
from clustercontrast.loss.triplet import SoftTripletLoss_vallia
from torch import nn


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None, cam=None, args=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.cam = cam
        self.clsLoss = nn.CrossEntropyLoss()
        self.softTripletLoss = SoftTripletLoss_vallia(margin=0.0).cuda()
        self.args = args
        self.lossList = {}
        self.labelList = {}


    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        print('lamb is ', self.args.lamb)
        end = time.time()

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            outputs = self._forward(inputs)

            loss_infoNce = self.memory(outputs, labels, epoch)
            loss_softTriplet_list = [self.softTripletLoss(output, output, labels) for output in outputs[1:6]]
            loss_softTriplet = sum(loss_softTriplet_list) / len(loss_softTriplet_list)

            cls_loss_list = [self.clsLoss(output, labels) for output in outputs[6:]]
            cls_loss = sum(cls_loss_list) / len(cls_loss_list)

            if self.args is not None:

                loss = self.args.lamb * loss_infoNce + (1 - self.args.lamb) * (loss_softTriplet + cls_loss)
            else:
                loss = loss_infoNce + loss_softTriplet + cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()
            BoardWriter.boardWriter.add_scalar('loss_infoNce', loss_infoNce, epoch)
            BoardWriter.boardWriter.add_scalar('loss_softTriplet', loss_softTriplet, epoch)
            BoardWriter.boardWriter.add_scalar('cls_loss', cls_loss, epoch)
            BoardWriter.boardWriter.add_scalar('loss_result', loss, epoch)

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs, img_names=None, epoch=None):
        return self.encoder(inputs)
