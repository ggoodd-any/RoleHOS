# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random

from mmseg.registry import DATASETS
# from .custom import CustomDataset
from mmengine.dataset import BaseDataset, Compose
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class EgoHOSDataset(BaseSegDataset):
    """EgoHOS dataset.

    Args:
        split (str): Split txt file for EgoHOS.
    """
    
    # CLASSES = ('background', 'aeroplane')

    # PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    #            [128, 0, 128]]

    METAINFO = dict(
        # classes = ('BG', 'Left_Object1', 'Right_Object1', 'Two_Object1'),   
        # palette=[[0, 0, 0], [255, 0, 255], [0, 255, 255], [0, 255, 0]],
        classes = ('background', 'Left_Hand', 'Right_Hand', \
               'Left_Object1', 'Right_Object1', 'Two_Object1', \
               'Left_Object2', 'Right_Object2', 'Two_Object2'),   
        palette=[[0, 0, 0], [255, 0, 0], [0, 0, 255], \
               [255, 0, 255], [0, 255, 255], [0, 255, 0], \
               [255, 204, 255], [204, 255, 255], [204, 255, 204]],
)

    # CLASSES = ('background', 'Left_Hand', 'Right_Hand', \
    #            'Left_Object1', 'Right_Object1', 'Two_Object1', \
    #            'Left_Object2', 'Right_Object2', 'Two_Object2')

    # PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255], \
    #            [255, 0, 255], [0, 255, 255], [0, 255, 0], \
    #            [255, 204, 255], [204, 255, 255], [204, 255, 204]]

    def __init__(self, **kwargs):
        super(EgoHOSDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        
    # def __init__(self,
    #              ann_file: str='',  # ann_file for obj
    #              img_suffix='.jpg',
    #              seg_map_suffix='.png',**kwargs) -> None:


@DATASETS.register_module()
class RandomLabelPairShuffleEgoHOSDataset(EgoHOSDataset):
    """EgoHOS dataset with image-label pair shuffle for shortcut audits."""

    def __init__(self, shuffle_seed: int = 17, **kwargs):
        self.shuffle_seed = int(shuffle_seed)
        super().__init__(**kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()
        if len(data_list) <= 1:
            return data_list
        seg_paths = [item.get('seg_map_path') for item in data_list]
        original_stems = [
            osp.splitext(osp.basename(item['img_path']))[0]
            for item in data_list
        ]
        shuffled = list(seg_paths)
        rng = random.Random(self.shuffle_seed)
        rng.shuffle(shuffled)
        for idx in range(len(shuffled)):
            if original_stems[idx] != osp.splitext(osp.basename(shuffled[idx]))[0]:
                continue
            swap_idx = (idx + 1) % len(shuffled)
            if swap_idx == idx:
                continue
            shuffled[idx], shuffled[swap_idx] = shuffled[swap_idx], shuffled[idx]
        if any(
                original_stems[idx] == osp.splitext(osp.basename(seg_path))[0]
                for idx, seg_path in enumerate(shuffled)):
            # Deterministic derangement fallback for pathologically ordered data.
            shuffled = seg_paths[1:] + seg_paths[:1]
        for item, seg_path in zip(data_list, shuffled):
            item['original_seg_map_path'] = item.get('seg_map_path')
            item['seg_map_path'] = seg_path
            item['random_label_pair_shuffle'] = True
            item['shuffle_seed'] = self.shuffle_seed
        return data_list
