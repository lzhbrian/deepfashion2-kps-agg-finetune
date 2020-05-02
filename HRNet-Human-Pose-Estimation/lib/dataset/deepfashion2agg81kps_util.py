"""
brief: aggregate original 294 kps to 81 kps
author: lzhbrian
date: 2020.3.31
"""

category_name = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
                 'long_sleeved_outwear', 'vest', 'sling',
                 'shorts', 'trousers', 'skirt',
                 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress']
num_kps_category = [25, 33, 31, 39, 15, 15, 10, 14, 8, 29, 37, 19, 19]
keypoint_name = []
for i in range(13):
    for j in range(num_kps_category[i]):
        keypoint_name.append(category_name[i] + '_' + str(j + 1))


def get_kpsname_to_id_map():

    kpsname_to_id_map = {}

    ## 领子
    # {1,2,3,4,5,6}: 领子的一圈 除了两个outwear, shorts, trousers, skirt之外都适用
    for category in ['short_sleeved_shirt', 'long_sleeved_shirt',
                     'vest', 'sling',
                     'short_sleeved_dress', 'long_sleeved_dress',
                     'vest_dress', 'sling_dress']:
        for i in [1,2,3,4,5,6]:
            kpsname_to_id_map[f'{category}_{i}'] = i
    for category in ['short_sleeved_outwear',
                     'long_sleeved_outwear']:
        for i in [1,2,3,5,6]:
            kpsname_to_id_map[f'{category}_{i}'] = i
    # {7,8}: 对于outwear来说，他的领子那两个点要另外计算
    kpsname_to_id_map[f'short_sleeved_outwear_4'] = 7
    kpsname_to_id_map[f'short_sleeved_outwear_26'] = 8
    kpsname_to_id_map[f'long_sleeved_outwear_4'] = 7
    kpsname_to_id_map[f'long_sleeved_outwear_34'] = 8

    ## 短袖 (左)
    # {9,10,11,12,13,14}: short_sleeved_shirt, short_sleeved_outwear, short_sleeved_dress
    for category in ['short_sleeved_shirt',
                     'short_sleeved_outwear',
                     'short_sleeved_dress']:
        for i in [7,8,9,10,11,12]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 2

    ## 短袖 (右)
    # {15,16,17,18,19,20}
    for category in ['short_sleeved_shirt',
                     'short_sleeved_outwear']:
        for i in [20,21,22,23,24,25]:
            kpsname_to_id_map[f'{category}_{i}'] = i - 5
    for i in [24,25,26,27,28,29]:
            kpsname_to_id_map[f'short_sleeved_dress_{i}'] = i - 9

    ## 长袖（左）
    # {21,22,23,24,25,26,27,28,29,30}
    for category in ['long_sleeved_shirt',
                     'long_sleeved_outwear',
                     'long_sleeved_dress']:
        for i in [7,8,9,10,11,12,13,14,15,16]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 14

    ## 长袖 (右)
    # {31,32,33,34,35,36,37,38,39,40}
    for category in ['long_sleeved_shirt',
                     'long_sleeved_outwear']:
        for i in [24,25,26,27,28,29,30,31,32,33]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 7
    for i in [28,29,30,31,32,33,34,35,36,37]:
            kpsname_to_id_map[f'long_sleeved_dress_{i}'] = i + 3

    ## 身体左半
    # {41,42,43}
    for category in ['short_sleeved_shirt',
                     'short_sleeved_outwear',
                     'short_sleeved_dress']:
        for i in [13,14,15]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 28
    for category in ['long_sleeved_shirt',
                     'long_sleeved_outwear',
                     'long_sleeved_dress']:
        for i in [17,18,19]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 24
    for category in ['vest', 'sling', 'vest_dress', 'sling_dress']:
        for i in [8,9,10]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 33

    ## 身体右半
    # {44,45,46}
    for category in ['short_sleeved_shirt',
                     'short_sleeved_outwear']:
        for i in [17,18,19]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 27
    for category in ['long_sleeved_shirt',
                     'long_sleeved_outwear',
                     'short_sleeved_dress']:
        for i in [21,22,23]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 23
    for i in [25,26,27]:
        kpsname_to_id_map[f'long_sleeved_dress_{i}'] = i + 19
    for category in ['vest', 'sling']:
        for i in [12,13,14]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 32
    for category in ['vest_dress', 'sling_dress']:
        for i in [16,17,18]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 28

    ## center
    # {47}
    kpsname_to_id_map['short_sleeved_shirt_16'] = 47
    kpsname_to_id_map['long_sleeved_shirt_20'] = 47
    kpsname_to_id_map['vest_11'] = 47
    kpsname_to_id_map['sling_11'] = 47

    ## outwear 中间部分
    # {48,49,50,51,52,53}
    for i in [27,28,29,30,31]:
        kpsname_to_id_map[f'short_sleeved_outwear_{i}'] = i + 21
    kpsname_to_id_map['short_sleeved_outwear_16'] = 53
    for i in [35,36,37,38,39]:
        kpsname_to_id_map[f'long_sleeved_outwear_{i}'] = i + 13
    kpsname_to_id_map['long_sleeved_outwear_20'] = 53

    ## vest, sling, vest_dress, sling_dress的肩点
    # {54,55,56,57}
    kpsname_to_id_map['vest_7'] = 54
    kpsname_to_id_map['vest_15'] = 55
    kpsname_to_id_map['vest_dress_7'] = 54
    kpsname_to_id_map['vest_dress_19'] = 55
    kpsname_to_id_map['sling_7'] = 56
    kpsname_to_id_map['sling_15'] = 57
    kpsname_to_id_map['sling_dress_7'] = 56
    kpsname_to_id_map['sling_dress_19'] = 57

    ## 裙摆
    # {58,59,60,61,62}
    for i in [16,17,18,19,20]:
        kpsname_to_id_map[f'short_sleeved_dress_{i}'] = i + 42
    for i in [20,21,22,23,24]:
        kpsname_to_id_map[f'long_sleeved_dress_{i}'] = i + 38
    for i in [11,12,13,14,15]:
        kpsname_to_id_map[f'vest_dress_{i}'] = i + 47
    for i in [11,12,13,14,15]:
        kpsname_to_id_map[f'sling_dress_{i}'] = i + 47

    ## 下半身的腰
    # {63,64,65}
    for category in ['shorts', 'trousers', 'skirt']:
        for i in [1,2,3]:
            kpsname_to_id_map[f'{category}_{i}'] = i + 62

    ## 裤腿
    # {66,67,68,69}
    kpsname_to_id_map['shorts_5'] = 66
    kpsname_to_id_map['shorts_6'] = 67
    kpsname_to_id_map['shorts_8'] = 68
    kpsname_to_id_map['shorts_9'] = 69

    kpsname_to_id_map['trousers_6'] = 66
    kpsname_to_id_map['trousers_7'] = 67
    kpsname_to_id_map['trousers_11'] = 68
    kpsname_to_id_map['trousers_12'] = 69

    ## 裤子中间点
    # {70}
    kpsname_to_id_map['shorts_7'] = 70
    kpsname_to_id_map['trousers_9'] = 70

    ## 裤子左右两侧中点
    # {71,72}
    kpsname_to_id_map['shorts_4'] = 71
    kpsname_to_id_map['shorts_10'] = 72
    kpsname_to_id_map['trousers_4'] = 71
    kpsname_to_id_map['trousers_14'] = 72

    ## 长裤的另外四个点
    # {73,74,75,76}
    kpsname_to_id_map['trousers_5'] = 73
    kpsname_to_id_map['trousers_8'] = 74
    kpsname_to_id_map['trousers_10'] = 75
    kpsname_to_id_map['trousers_13'] = 76

    ## 短裙的另外五个点
    # {77,78,79,80,81}
    kpsname_to_id_map['skirt_4'] = 77
    kpsname_to_id_map['skirt_5'] = 78
    kpsname_to_id_map['skirt_6'] = 79
    kpsname_to_id_map['skirt_7'] = 80
    kpsname_to_id_map['skirt_8'] = 81

    return kpsname_to_id_map


kpsname_to_id_map = get_kpsname_to_id_map()
ori294_to_agg81kps_map = {}
for idx, n in enumerate(keypoint_name):
    ori294_to_agg81kps_map[idx] = kpsname_to_id_map[n] - 1


# input: new_kps_list
# output: ori_kps_list
def convert_back(new_kps_list):
    if len(new_kps_list) == 294 * 3:
        return new_kps_list
    ori_kps_list = [0] * 294 * 3
    for ori_idx in range(294):
        new_idx = ori294_to_agg81kps_map[ori_idx]
        ori_kps_list[ori_idx * 3 + 0] = new_kps_list[new_idx * 3 + 0]
        ori_kps_list[ori_idx * 3 + 1] = new_kps_list[new_idx * 3 + 1]
        ori_kps_list[ori_idx * 3 + 2] = new_kps_list[new_idx * 3 + 2]
    return ori_kps_list


# flip map for agg81kps
def get_deepfashion2ahh81kps_flip_map():

    kpsname_to_id_map = get_kpsname_to_id_map()
    keypoint_flip_map_info = {
        'short_sleeved_shirt': [(2,6), (3,5), (7,25), (8,24), (9,23),
                                (10,22), (11,21), (12,20), (13,19), (14,18),
                                (15,17)],
        'long_sleeved_shirt': [(2,6), (3,5), (7,33), (8,32), (9,31),
                               (10,30), (11,29), (12,28), (13,27), (14,26),
                               (15,25), (16,24), (17,23), (18,22), (19,21)],
        'short_sleeved_outwear': [(2,6), (3,5), (4,26), (7,25), (8,24),
                                  (9,23), (10,22), (11,21), (12,20), (13,19),
                                  (14,18), (15,17), (16,29), (31,28), (30,27)],
        'long_sleeved_outwear': [(4,34), (3,5), (2,6), (7,33), (8,32),
                                 (9,31), (10,30), (11,29), (12,28), (13,27),
                                 (14,26), (15,25), (16,24), (17,23), (18,22),
                                 (19,21), (20,37), (39,36), (38,35)],
        'vest': [(3,5), (2,6), (7,15), (8,14), (9,13),
                 (10,12)],
        'sling': [(3,5), (2,6), (7,15), (8,14), (9,13),
                  (10,12)],
        'shorts': [(1,3), (4,10), (5,9), (6,8)],
        'trousers': [(1,3), (4,14), (5,13), (6,12), (7,11), (8,10)],
        'skirt': [(1,3), (4,8), (5,7)],
        'short_sleeved_dress': [(3,5), (2,6), (7,29), (8,28), (9,27),
                                (10,26), (11,25), (12,24), (13,23), (14,22),
                                (15,21), (16,20), (17,19)],
        'long_sleeved_dress': [(3,5), (2,6), (7,37), (8,36), (9,35),
                               (10,34), (11,33), (12,32), (13,31), (14,30),
                               (15,29), (16,28), (17,27), (18,26), (19,25),
                               (20,24), (21,23)],
        'vest_dress': [(3,5), (2,6), (7,19), (8,18), (10,16),
                       (11,15), (12,14)],
        'sling_dress': [(3,5), (2,6), (7,19), (8,18), (10,16),
                       (11,15), (12,14)]
    }
    keypoint_flip_map = []
    for cate, pairs in keypoint_flip_map_info.items():
        for p in pairs:
            a = cate + '_' + str(p[0])
            b = cate + '_' + str(p[1])
            keypoint_flip_map.append((a,b))

    keypoint_flip_map_idx = []
    for pair in keypoint_flip_map:
        a = kpsname_to_id_map[pair[0]] - 1
        b = kpsname_to_id_map[pair[1]] - 1
        if [a, b] not in keypoint_flip_map_idx:
            keypoint_flip_map_idx.append([a,b])

    return keypoint_flip_map, keypoint_flip_map_idx
