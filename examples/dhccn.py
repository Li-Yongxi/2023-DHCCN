# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import random
import sys,datetime

import numpy as np

sys.path.append(".")
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
from clustercontrast.utils.board_writter import BoardWriter

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.evaluators import Evaluator, extract_features, conf_eval_gauss
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.models.cm_dhccn import ClusterMemory
import matplotlib

from clustercontrast.utils.env_utils import EnvDict
import fitlog
from torch.distributions.multivariate_normal import MultivariateNormal

fitlog.set_log_dir("fitlog_logs/")


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=2048, num_classes=100)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    fitlog.commit(__file__)

    args = parser.parse_args()
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%m%d%H%M')
    args.logs_dir= osp.join(args.logs_dir , str(timestamp)+'_'+args.dataset + '_' + args.arch
                            + '_l' + str(args.lamb)) + '_m' + str(args.momentum) + '_e' + str(args.eps)
    EnvDict.set_value(EnvDict, 'isEvalCam', False)
    EnvDict.set_value(EnvDict, 'args', args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        cudnn.deterministic = True

    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    main_worker(args)



def main_worker(args):
    global start_epoch, best_mAP
    best_model_cluster_num = 0
    start_time = time.monotonic()

    # cudnn.benchmark = True
    cudnn.benchmark = False

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    print("==========\nArgs:{}\n==========".format(args))
    BoardWriter.setWriter(BoardWriter, args.logs_dir.replace('logs', 'runs'))
    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)
    criterion_CE = nn.CrossEntropyLoss(reduction='none')
    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]

    trainer = ClusterContrastTrainer(model, cam=None, args=args)

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    pseudo_labels = None

    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, features_p1, features_p2, _, l_h, l_g, l_l = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            features_p1 = torch.cat([features_p1[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            features_p2 = torch.cat([features_p2[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            l_h = torch.cat([l_h[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            l_g = torch.cat([l_g[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            l_l = torch.cat([l_l[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)


            prob_full = None
            distribution_h_dict = {}
            distribution_g_dict = {}
            distribution_l_dict = {}

            if epoch>0:
                ind = (pseudo_labels >= 0)
                prob_full = torch.ones(len(pseudo_labels), dtype=torch.float32) * 0.1
                p = pseudo_labels[ind]

                loss = criterion_CE(l_h[ind], torch.from_numpy(p.astype(int)))

                loss = (loss - loss.min()) / (loss.max() - loss.min())
                loss_re = loss.reshape(-1, 1)

                prob = conf_eval_gauss(loss_re)
                prob_full[ind] = torch.from_numpy(prob).float()
                BoardWriter.boardWriter.add_histogram('prob', prob_full, epoch)

                print("calibration begin")
                for p in set(pseudo_labels):
                    if p == -1:
                        continue

                    p_index = (pseudo_labels == p)
                    noise_index = torch.from_numpy(p_index) & (prob_full < args.noisy_threshold)
                    clean_index = torch.from_numpy(p_index) & (prob_full >= args.noisy_threshold)
                    sp = prob_full[noise_index]
                    dp = 1 - prob_full[noise_index]
                    f_h = features[clean_index]
                    f_g = features_p1[clean_index]
                    f_l = features_p2[clean_index]

                    if len(f_h) < 2:
                        continue

                    num = len(dp)
                    if num == 0:
                        continue
                        print("num is 0")

                    c_h = torch.cov(f_h.T)
                    c_g = torch.cov(f_g.T)
                    c_l = torch.cov(f_l.T)

                    distribution_h_dict[p] = (torch.mean(f_h, axis=0), torch.mm(c_h, c_h.T).add_(torch.eye(2048)))
                    distribution_g_dict[p] = (torch.mean(f_g, axis=0), torch.mm(c_g, c_g.T).add_(torch.eye(2048)))
                    distribution_l_dict[p] = (torch.mean(f_l, axis=0), torch.mm(c_l, c_l.T).add_(torch.eye(2048)))
                    try:
                        features[noise_index] = torch.mul(features[noise_index].t(),sp).t() + \
                                            torch.mul(MultivariateNormal(distribution_h_dict[p][0], distribution_h_dict[p][1]).sample().repeat(num).reshape(num, -1).t(), dp).t()
                        features_p1[noise_index] = torch.mul(features_p1[noise_index].t(), sp).t() + \
                                               torch.mul(MultivariateNormal(distribution_g_dict[p][0], distribution_g_dict[p][1]).sample().repeat(num).reshape(num, -1).t(), dp).t()
                        features_p2[noise_index] = torch.mul(features_p2[noise_index].t(), sp).t() + \
                                               torch.mul(MultivariateNormal(distribution_l_dict[p][0], distribution_l_dict[p][1]).sample().repeat(num).reshape(num, -1).t() , dp).t()
                    except:
                        print()
                print("calibration done")


            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

            print("epoch: {} \n pseudo_labels: {} \n num_cluster: {}".format(epoch, pseudo_labels.tolist()[:10],
                                                                             num_cluster))
            torch.save(pseudo_labels, osp.join(args.logs_dir, "epoch_" + str(epoch) + "_d_pseudo_labels" + ".npy"))


        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features, features_p1,features_p2, prob):
            centers = collections.defaultdict(list)
            centers_p1 = collections.defaultdict(list)
            centers_p2 = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue

                if prob is not None:
                    centers[labels[i]].append(features[i]*prob[i])
                    centers_p1[labels[i]].append(features_p1[i]*prob[i])
                    centers_p2[labels[i]].append(features_p2[i]*prob[i])
                else:
                    centers[labels[i]].append(features[i])
                    centers_p1[labels[i]].append(features_p1[i])
                    centers_p2[labels[i]].append(features_p2[i])
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers_p1 = [
                torch.stack(centers_p1[idx], dim=0).mean(0) for idx in sorted(centers_p1.keys())
            ]
            centers_p2 = [
                torch.stack(centers_p2[idx], dim=0).mean(0) for idx in sorted(centers_p2.keys())
            ]

            centers = torch.stack(centers, dim=0)
            centers_p1 = torch.stack(centers_p1, dim=0)
            centers_p2 = torch.stack(centers_p2, dim=0)
            return centers, centers_p1, centers_p2

        cluster_features, cluster_features_p1, cluster_features_p2, = \
            generate_cluster_features(pseudo_labels, features, features_p1, features_p2, prob_full)

        # del cluster_loader, features
        del features, features_p1, features_p2


        model.module._reset_fc(num_cluster)



        memory = ClusterMemory(2048, num_cluster, temp=args.temp,
                                    momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory.predict_features = F.normalize(cluster_features, dim=1).cuda()
        memory.global_p1_features = F.normalize(cluster_features_p1, dim=1).cuda()
        memory.global_p2_features = F.normalize(cluster_features_p2, dim=1).cuda()
        if epoch is not 0:
            memory.prob = prob_full.cuda()


        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False, args=args)
            fitlog.add_metric({"dev": {"mAP": mAP}}, step=epoch)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'num_cluster': num_cluster
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_ep'+ str(epoch) +'.pth.tar'))
            if is_best:
                best_model_cluster_num = num_cluster
                fitlog.add_best_metric({"dev": {"mAP": mAP}}, name=args.dataset)
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
            BoardWriter.boardWriter.add_scalar('mAP', mAP, epoch)
        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module._reset_fc(best_model_cluster_num)
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, args=args)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))
    fitlog.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")

    parser.add_argument('--lr', type=float, default=0.00035,
                        # parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--noisy-threshold', type=float, default=0.1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")


    main()
