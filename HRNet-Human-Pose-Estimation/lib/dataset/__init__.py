# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .deepfashion2 import Deepfashion2Dataset as deepfashion2
from .deepfashion2agg81kps import Deepfashion2Agg81KpsDataset as deepfashion2agg81kps
from .deepfashion2agg81kpspercategory import Deepfashion2Agg81KpsDatasetPerCategory as deepfashion2agg81kpspercategory
