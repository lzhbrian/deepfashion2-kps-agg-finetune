import os
import sys

gpus = '0,1,2,3,4,5,6'
GPUS = '\'(0,1,2,3,4,5,6)\''

test_or_validation_set = 'validation'
test_or_validation_set = 'test'
flip_test = 'True'
# flip_test = 'False'
img_thre = '0.01'


def run_category_finetune_hflip(category_id):
    """trainhflip finetune model"""
    print(f'Running category{category_id} ...')
    det_file = f'../htcdet/htc-{test_or_validation_set}-det-result-3scale-hflip.bbox.json_' \
               f'{category_id}.json'
    cfg_file = f'experiments/deepfashion2/' \
               f'w48_512x384_adam_lr1e-3-agg81kps-category{category_id}-hflip.yaml'
    weight = f'output/deepfashion2agg81kpspercategory/pose_hrnet/' \
             f'w48_512x384_adam_lr1e-3-agg81kps-category{category_id}-hflip/model_best.pth'
    cmd = f'CUDA_VISIBLE_DEVICES={gpus} python tools/test.py' \
          f' --cfg {cfg_file}' \
          f' PRINT_FREQ 10' \
          f' GPUS {GPUS}' \
          f' TEST.MODEL_FILE {weight}' \
          f' TEST.USE_GT_BBOX False' \
          f' TEST.COCO_BBOX_FILE {det_file}' \
          f' TEST.IMAGE_THRE {img_thre}' \
          f' DATASET.TEST_SET {test_or_validation_set}' \
          f' TEST.FLIP_TEST {flip_test}' \
          f' TEST.BATCH_SIZE_PER_GPU 32'
    print(cmd)
    os.system(cmd)
    print(f'finished category{category_id} ...')


def run_category_finetune(category_id):
    """finetune model"""
    print(f'Running category{category_id} ...')
    det_file = f'../htcdet/htc-{test_or_validation_set}-det-result-3scale-hflip.bbox.json_' \
               f'{category_id}.json'
    cfg_file = f'experiments/deepfashion2/' \
               f'w48_512x384_adam_lr1e-3-agg81kps-category{category_id}.yaml'
    weight = f'output/deepfashion2agg81kpspercategory/pose_hrnet/' \
             f'w48_512x384_adam_lr1e-3-agg81kps-category{category_id}/model_best.pth'
    # weight = f'output/deepfashion2agg81kpspercategory/pose_hrnet/' \
    #          f'w48_512x384_adam_lr1e-3-agg81kps-category{category_id}/final_state.pth'
    cmd = f'CUDA_VISIBLE_DEVICES={gpus} python tools/test.py' \
          f' --cfg {cfg_file}' \
          f' PRINT_FREQ 10' \
          f' GPUS {GPUS}' \
          f' TEST.MODEL_FILE {weight}' \
          f' TEST.USE_GT_BBOX False' \
          f' TEST.COCO_BBOX_FILE {det_file}' \
          f' TEST.IMAGE_THRE {img_thre}' \
          f' DATASET.TEST_SET {test_or_validation_set}' \
          f' TEST.FLIP_TEST {flip_test}' \
          f' TEST.BATCH_SIZE_PER_GPU 32'
    print(cmd)
    os.system(cmd)
    print(f'finished category{category_id} ...')


if __name__ == '__main__':
    # for i in range(1, 14):
    # for i in [3,6]:
    for i in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
        # run_category_finetune_hflip(i)
        run_category_finetune(i)
