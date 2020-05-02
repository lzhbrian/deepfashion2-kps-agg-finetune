import os
import sys
import json


test_or_validation_set = 'validation'
test_or_validation_set = 'test'
o_folder = f'./kpsresult-{test_or_validation_set}-finetune-trainhflip-ttahflip/'
o_folder = f'./kpsresult-{test_or_validation_set}-finetune-ttahflip/'
os.makedirs(o_folder, exist_ok=True)


overall_d = []


def collect_category(category_id):
    print(f'processing {category_id}...')

    r_file = f'output/deepfashion2agg81kpspercategory/pose_hrnet/' \
             f'w48_512x384_adam_lr1e-3-agg81kps-category{category_id}-hflip/results/' \
             f'keypoints_{test_or_validation_set}_results_0.json'
    r_file = f'output/deepfashion2agg81kpspercategory/pose_hrnet/' \
        f'w48_512x384_adam_lr1e-3-agg81kps-category{category_id}/results/' \
        f'keypoints_{test_or_validation_set}_results_0.json'

    d = json.load(open(r_file))
    overall_d.extend(d)

    cmd = f'cp {r_file} {o_folder}/{category_id}.json'
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    # for i in [3,6]:
    for i in range(1, 14):
        collect_category(i)

    with open(f'{o_folder}/overall.json', 'w') as fp:
        print(json.dumps(overall_d), file=fp)
