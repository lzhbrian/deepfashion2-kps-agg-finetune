"""
融合两个kps结果，需要具有同样的det result
usage:
    python ensemble.py r1.json r2.json o_file.json
"""

from tqdm import tqdm
import json
import copy
import sys
import numpy as np
from collections import defaultdict


def ensemble(j_list):
    """
    :param r_list: list of result list
    :return: ensembled list
    """

    num_models = len(j_list)
    all_r = []

    for j in j_list:
        r = defaultdict(lambda: {})
        for inst in tqdm(j):
            identifier = inst['image_id'] * 100 + inst['category_id']
            r[identifier][inst['center'][0] + inst['center'][1] * 1.7] = copy.deepcopy(inst)
        all_r.append(r)

    final_result = []

    # 遍历所有det
    for idx in tqdm(range(len(j_list[0]))):

        identifier = j_list[0][idx]['image_id'] * 100 + j_list[0][idx]['category_id']
        center_k = j_list[0][idx]['center'][0] + j_list[0][idx]['center'][1] * 1.7

        ensemble_inst = copy.deepcopy(j_list[0][idx])
        ensemble_inst['score'] = 0
        ensemble_inst['keypoints'] = np.array([0.0] * 294 * 3)

        # 遍历所有model
        for model_id in range(num_models):
            ensemble_inst['keypoints'] += np.array(all_r[model_id][identifier][center_k]['keypoints'])
            ensemble_inst['score'] += np.array(all_r[model_id][identifier][center_k]['score'])

        ensemble_inst['keypoints'] = list(ensemble_inst['keypoints'] / num_models)
        ensemble_inst['score'] = ensemble_inst['score'] / num_models
        final_result.append(ensemble_inst)

    return final_result


if __name__ == '__main__':
    file_1 = sys.argv[1]
    file_2 = sys.argv[2]
    o_file = sys.argv[3]

    j1 = json.load(open(file_1))
    j2 = json.load(open(file_2))

    e = ensemble([j1, j2])

    with open(o_file, 'w') as fp:
        print(json.dumps(e), file=fp)
