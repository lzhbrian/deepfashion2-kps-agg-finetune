"""
brief: aggregate original 294 kps to 81 kps
author: lzhbrian
date: 2020.3.31
usage: 
    python scripts/dataset_agg81kps.py
"""

from tqdm import tqdm
import copy
import json
import numpy as np

import sys
sys.path.insert(0, '../dataset/')
from lib.dataset import deepfashion2agg81kps_util


def agg81kps(infile, outfile):
    with open(infile) as f:
        d = json.load(f)
    new_annotations = []
    for ann in tqdm(d['annotations']):
        new_ann = copy.deepcopy(ann)
        ori_kps_list = ann['keypoints']
        # input: kps_list
        # output: new_kps_list
        # [x1,y1,s1, x2,y2,s2, ...]
        new_kps_list = [0.0] * 81 * 3
        for ori_idx in range(294):
            new_idx = deepfashion2agg81kps_util.ori294_to_agg81kps_map[ori_idx]
            # print(ori_idx, new_idx, ori_kps_list[ori_idx * 3 + 0:ori_idx * 3 + 3])
            if ori_kps_list[ori_idx * 3 + 2] > 0:
                if new_kps_list[new_idx * 3 + 2] != 0:
                    print(ann)
                    break
                new_kps_list[new_idx * 3 + 0] = ori_kps_list[ori_idx * 3 + 0]
                new_kps_list[new_idx * 3 + 1] = ori_kps_list[ori_idx * 3 + 1]
                new_kps_list[new_idx * 3 + 2] = ori_kps_list[ori_idx * 3 + 2]
        # print(np.sum(np.array(new_kps_list) > 0), np.sum(np.array(ori_kps_list) > 0))
        assert(np.sum(np.array(new_kps_list) > 0) == np.sum(np.array(ori_kps_list) > 0))
        new_ann['keypoints'] = new_kps_list
        new_annotations.append(new_ann)
    new_d = copy.deepcopy(d)
    new_d['annotations'] = new_annotations
    def cvt_c(cate):
        return_cate = copy.deepcopy(cate)
        return_cate['keypoints'] = [str(i) for i in range(1, 82)]
        return return_cate
    new_categories = [cvt_c(c) for c in d['categories']]
    new_d['categories'] = new_categories
    with open(outfile, 'w') as fp:
        print(json.dumps(new_d), file=fp)


if __name__ == '__main__':
    infile = './data/deepfashion2/annotations/train_coco.json'
    outfile = './data/deepfashion2/annotations/train_coco_agg81kps.json'
    print(f'Processing training set, in:{infile}, out:{outfile}')
    agg81kps(infile, outfile)

    infile = './data/deepfashion2/annotations/validation_coco.json'
    outfile = './data/deepfashion2/annotations/validation_coco_agg81kps.json'
    print(f'Processing validation set, in:{infile}, out:{outfile}')
    agg81kps(infile, outfile)


