from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from ..utils import to_numpy
from clustercontrast.utils.env_utils import EnvDict
import os.path as osp
import os
import shutil
import torchvision, torch


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None, gallery_fp = None, query_fp= None):

    args = EnvDict.get_value(EnvDict, 'args')
    if EnvDict.get_value(EnvDict, 'isEvalCam'):
        run_dir = args.logs_dir.replace('logs', 'runs')
        # run_dir = args["logs_dir"].replace('logs', 'runs')
        run_img_dir = osp.join(run_dir, "map")
        print(run_img_dir)
        if ~(osp.exists(run_img_dir) and osp.isdir(run_img_dir)):
            os.makedirs(osp.join(run_img_dir, str(EnvDict.get_value(EnvDict, 'EPOCH'))), exist_ok=True)
            print(osp.join(run_img_dir, str(EnvDict.get_value(EnvDict, 'EPOCH'))))
        run_img_dir = osp.join(run_img_dir, str(EnvDict.get_value(EnvDict, 'EPOCH')))


    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    if EnvDict.get_value(EnvDict, 'isEvalCam') and gallery_fp is not None:
        gallery_fp = np.array(gallery_fp)
        matches_img = gallery_fp[indices]
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))

        if EnvDict.get_value(EnvDict, 'isEvalCam') and gallery_fp is not None and query_fp is not None:
            y_img = matches_img[i, valid]
            # query_galleryTop10 = []
            # query_img = Image.open(query_fp[i]).convert('RGB').resize((128,256), Image.ANTIALIAS)
            # query_galleryTop10.append(np.transpose(np.array(query_img), (2,0,1)))
            path = osp.join(run_img_dir+"/"+str(i))
            os.makedirs(path, exist_ok=True)
            shutil.copy(query_fp[i], osp.join(path, "query_"+osp.basename(query_fp[i])))
            with open(osp.join(run_img_dir,str(i))+".txt", 'w', encoding="utf-8") as f:
                f.write("query is" + str(query_fp[i]) + "\n")
                f.write("top 10\n")

                for j in range(10):
                    # img = Image.open(y_img[j]).convert('RGB')
                    # img_tensor = np.transpose(np.array(img.resize((128,256), Image.ANTIALIAS)), (2,0,1))
                    # query_galleryTop10.append(img_tensor)
                    f.write(str(j) + ":\t" + y_img[j] + "\t" + str(y_score[j])+"\n")
                    shutil.copy(y_img[j], osp.join(path, "gallery_"+ str(j) + "_" + osp.basename(y_img[j])))
                # grid = torchvision.utils.make_grid(torch.from_numpy(np.array(query_galleryTop10)), scale_each =True)

                # torchvision.utils.save_image(torch.from_numpy(np.array(query_galleryTop10))/255.0, osp.join(run_img_dir,str(i) + ".jpg"))

    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)
