from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch
from sklearn.mixture import GaussianMixture

def extract_cnn_feature(model, inputs, fnames=None):
    # inputs = to_torch(inputs).cuda()
    inputs = to_torch(inputs).cuda()
    if fnames is None:
        outputs = model(inputs)
    else:
        outputs = model(inputs, img_names=fnames)
    outputs = [outputs[i].data.cpu() for i in range(len(outputs))]

    return outputs[0], outputs

def conf_eval_kl(f1, f2):
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)
    variance = torch.sum(kl_distance(log_sm(f1), sm(f2.detach())), dim=1)
    # variance = torch.exp(-variance) + 1
    return variance

def conf_eval_gauss(loss):
    gmm_V = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    gmm_V.fit(loss)
    prob_V = gmm_V.predict_proba(loss)
    prob_V = prob_V[:, gmm_V.means_.argmin()]

    return prob_V


def extract_features(model, data_loader, print_freq=50, isCam=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    features_p1 = OrderedDict()
    features_p2 = OrderedDict()
    l_h = OrderedDict()
    l_g = OrderedDict()
    l_l = OrderedDict()
    labels = OrderedDict()


    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs, full_outputs = extract_cnn_feature(model, imgs)

            for fname, output, output_p1, output_p2, cls_h, cls_g, cls_l, pid in  zip(fnames, outputs, full_outputs[1], full_outputs[2], full_outputs[6], full_outputs[7], full_outputs[8], pids):
                features[fname] = output
                features_p1[fname] = output_p1
                features_p2[fname] = output_p2
                l_h[fname] = cls_h
                l_g[fname] = cls_g
                l_l[fname] = cls_l

                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, features_p1,features_p2, labels, l_h, l_g, l_l



def fliphor(inputs):
    inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
    return inputs.index_select(3,inv_idx)



def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
        query_fp = [item[0] for item in query]
        gallery_fp = [item[0] for item in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, gallery_fp = gallery_fp, query_fp = query_fp)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False, args=None):

        features, _, _, _, _, _, _ = extract_features(self.model, data_loader)
        # features, _ = extract_feature_3(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)