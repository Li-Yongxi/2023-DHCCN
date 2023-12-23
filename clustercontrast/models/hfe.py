import copy

import torch
from torch import nn
import torch.nn.functional as F
from .pooling import build_pooling_layer
from torchvision.models.resnet import resnet50, Bottleneck
from clustercontrast.utils.board_writter import BoardWriter

class HFE(nn.Module):
    def __init__(self, num_features=0, num_classes=0, heat_map=False):

        super(HFE, self).__init__()
        self.num_features = num_features
        self.heat_map=heat_map
        resnet = resnet50(pretrained=True)

        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.gap = build_pooling_layer("gem")
        self.w = nn.Parameter(torch.ones(3))
        self.pw = nn.Parameter(torch.ones(2))
        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        pool2d = nn.AvgPool2d
        self.pool_zg_p1 = build_pooling_layer("gem")
        self.pool_zg_p3 = build_pooling_layer("gem")
        self.pool_zp3 = pool2d(kernel_size=(8, 8))


        feat_bn = nn.BatchNorm1d(2048)
        feat_bn.bias.requires_grad_(False)
        nn.init.constant_(feat_bn.weight, 1)
        nn.init.constant_(feat_bn.bias, 0)


        self.feat_bn_0 = copy.deepcopy(feat_bn)
        self.feat_bn_1 = copy.deepcopy(feat_bn)
        self.feat_bn_2 = copy.deepcopy(feat_bn)
        self.feat_bn_3 = copy.deepcopy(feat_bn)

        self.fc_id_2048_0 = nn.Linear(num_features, num_classes, bias=False)
        self.fc_id_2048_1 = nn.Linear(num_features, num_classes, bias=False)
        self.fc_id_h = nn.Linear(num_features, num_classes, bias=False)


        self.fc_id_256_1_0 = nn.Linear(num_features, num_classes, bias=False)
        self.fc_id_256_1_1 = nn.Linear(num_features, num_classes, bias=False)
        self.fc_id_256_1_2 = nn.Linear(num_features, num_classes, bias=False)



        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_1_2)
        


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')

    def _reset_fc(self, num_classes):
        self.num_classes = num_classes
        self.fc_id_2048_0 = nn.Linear(self.num_features, num_classes, bias=False).cuda()
        self.fc_id_2048_1 = nn.Linear(self.num_features, num_classes, bias=False).cuda()


        self.fc_id_256_1_0 = nn.Linear(self.num_features, num_classes, bias=False).cuda()
        self.fc_id_256_1_1 = nn.Linear(self.num_features, num_classes, bias=False).cuda()
        self.fc_id_256_1_2 = nn.Linear(self.num_features, num_classes, bias=False).cuda()
        self.fc_id_h = nn.Linear(self.num_features, num_classes, bias=False).cuda()

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_1_2)
        self._init_fc(self.fc_id_h)

    def forward(self, x):
        x = self.backone(x)
        p1 = self.p1(x)
        p3 = self.p3(x)
        zg_p1 = self.pool_zg_p1(p1)
        zp3 = self.pool_zp3(p3)

        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        fg_p1 = zg_p1

        f0_p3 = z0_p3
        f1_p3 = z1_p3
        f2_p3 = z2_p3

        fg_p1 = fg_p1.view(fg_p1.size(0), -1)

        f0_p3 = f0_p3.view(f0_p3.size(0), -1)
        f1_p3 = f1_p3.view(f1_p3.size(0), -1)
        f2_p3 = f2_p3.view(f2_p3.size(0), -1)

        fg_p1 = F.normalize(self.feat_bn_0(fg_p1))

        f0_p3 = F.normalize(self.feat_bn_1(f0_p3))
        f1_p3 = F.normalize(self.feat_bn_2(f1_p3))
        f2_p3 = F.normalize(self.feat_bn_3(f2_p3))

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))

        fg_p3 = torch.sum(torch.stack([w1 * f0_p3, w2 * f1_p3, w3 * f2_p3]), dim=0)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p3 = self.fc_id_2048_1(fg_p3)

        l0_p3 = self.fc_id_256_1_0(f0_p3)
        l1_p3 = self.fc_id_256_1_1(f1_p3)
        l2_p3 = self.fc_id_256_1_2(f2_p3)


        pw1 = torch.exp(self.pw[0]) / torch.sum(torch.exp(self.pw))
        pw2 = torch.exp(self.pw[1]) / torch.sum(torch.exp(self.pw))
        predict = torch.sum(torch.stack([pw1*fg_p1, pw2*fg_p3]), dim=0)

        l_h = self.fc_id_h(predict)
        return predict, fg_p1, fg_p3, f0_p3, f1_p3, f2_p3, l_h, l_p1, l_p3, l0_p3, l1_p3, l2_p3