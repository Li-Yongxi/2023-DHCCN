import collections
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd
from clustercontrast.utils.board_writter import BoardWriter

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        # outputs = inputs.mm(ctx.features.t())
        outputs = inputs.bmm(ctx.features.permute(0, 2, 1))

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.bmm(ctx.features)

        # momentum update
        for i, input in enumerate(inputs):
            for x, y in zip(input, targets):
                ctx.features[i][y] = ctx.momentum * ctx.features[i][y] + (1. - ctx.momentum) * x
                ctx.features[i][y] /= ctx.features[i][y].norm()

        return grad_inputs, None, None, None

class CM_Ori(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None

def cm(inputs, indexes, features, momentum=0.5):
    # return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
    return CM_Ori.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('predict_features', torch.zeros(num_samples, num_features))
        self.register_buffer('global_p1_features', torch.zeros(num_samples, num_features))
        self.register_buffer('global_p2_features', torch.zeros(num_samples, num_features))


    def cal_loss(self, inputs, targets, features):
        output = cm(inputs, targets, features, self.momentum)
        output /= self.temp
        loss = F.cross_entropy(output, targets)
        return loss

    def forward(self, inputs, targets, epoch=0):
        # inputs = F.normalize(inputs, dim=1).cuda()
        temp_inputs = []
        for i in range(0,3):
            temp_inputs.append(F.normalize(inputs[i], dim=1).cuda())

        loss_predict = self.cal_loss(temp_inputs[0], targets, self.predict_features)
        loss_p1_g = self.cal_loss(temp_inputs[1], targets, self.global_p1_features)
        
        loss_p2_l = self.cal_loss(temp_inputs[2], targets, self.global_p2_features)
        log_mean = ((temp_inputs[1].softmax(dim=-1) + temp_inputs[2].softmax(dim=-1)) / 2).log()
        loss_js = (F.kl_div(log_mean, temp_inputs[1].softmax(dim=-1), reduction='sum') + F.kl_div(log_mean, temp_inputs[
            2].softmax(dim=-1), reduction='sum')) / 2
        loss = loss_predict + loss_p1_g + loss_p2_l + loss_js

        BoardWriter.boardWriter.add_scalar('loss_predict', loss_predict, epoch)
        BoardWriter.boardWriter.add_scalar('loss_p1_g', loss_p1_g, epoch)
        BoardWriter.boardWriter.add_scalar('loss_p2_l', loss_p2_l, epoch)
        BoardWriter.boardWriter.add_scalar('loss_js', loss_js, epoch)

        """inputs = F.normalize(inputs, dim=1).cuda()

        outputs = cm(inputs, targets, self.global_p1_features, self.momentum)"""

        return loss



