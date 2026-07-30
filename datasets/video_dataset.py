import os
import torch
import numpy as np
import json
from datasets.test_dataset import TestDataset

decord_available = False
try:
    import decord
    decord.bridge.set_bridge('torch')
    decord_available = True
except ImportError:
    pass


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, **kwargs):
        self.num_frames = kwargs['min_num_frames']
        self.sample_size = kwargs['video_size']
        self.traj_txt_path = kwargs.get('traj_txt_path', None)
        self.relative_to_source = kwargs.get('relative_to_source', False)
        self.rotation_only = kwargs.get('rotation_only', False)
        self.adaptive_frame = kwargs.get('adaptive_frame', True)
        self.freeze_repeat = kwargs.get('freeze_repeat', 0)
        self.freeze_frame = kwargs.get('freeze_frame', None)

        if not isinstance(json_path, str):
            json_path = json_path[0]
        assert isinstance(json_path, str), f"json_path must be a string, got {type(json_path)}"

        self.metadata_list = json.load(open(json_path, 'r'))
        self.dataset = {entry['video_path']: entry for entry in self.metadata_list}
        for entry in self.metadata_list:
            entry['dataset_type'] = 'test'

        self.test_dataset = TestDataset(
            self.sample_size,
            self.num_frames,
            traj_txt_path=self.traj_txt_path,
            relative_to_source=self.relative_to_source,
            rotation_only=self.rotation_only,
            adaptive_frame=self.adaptive_frame,
            freeze_repeat=self.freeze_repeat,
            freeze_frame=self.freeze_frame,
        )
        print(f"Loaded {len(self.dataset)} videos.")

    def __getitem__(self, index):
        while True:
            try:
                source_data_key = list(self.dataset.keys())[index]
                data = self.test_dataset.get_data(self.dataset[source_data_key])
                data = self._temporal_sampling(data)
                data['index'] = index
                break
            except Exception as e:
                import traceback
                import random
                print("Error info:", e)
                traceback.print_exc()
                index = random.randrange(len(self.dataset))
        return data

    def _temporal_sampling(self, data):
        current_frames = data['source_video'].shape[0]
        target_frames = self.num_frames
        if target_frames < current_frames:
            indices = torch.linspace(0, current_frames - 1, target_frames).long()
            for key in (
                'source_video', 'target_video', 'render_video', 'mask_video',
                'source_c2w', 'target_c2w', 'target_extrinsics', 'reference_depth',
            ):
                if (
                    key in data
                    and isinstance(data[key], (torch.Tensor, np.ndarray))
                    and data[key].shape[0] == current_frames
                ):
                    data[key] = data[key][indices]
        return data

    def __len__(self):
        return len(self.dataset)
