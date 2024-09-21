import json

import numpy as np
import torch
import torch.utils.data as Data


class TranscriptionDataset(Data.Dataset):
    def __init__(
        self,
        manifest_path,
        slice_sec,
        token_rate,
    ) -> None:
        super().__init__()

        with open(manifest_path) as f:
            self.data = [json.loads(line) for line in f]
        
        self.rand_slice_window = slice_sec * token_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        output_data = dict()
        mert_feature = torch.from_numpy(np.load(self.data[idx]["mert_path"]))
        mert_feature = torch.squeeze(mert_feature, 0)
        label_feature = torch.from_numpy(np.load(self.data[idx]["label_path"]))
        start_index = np.random.randint(low=0, high=label_feature.shape[0]-self.rand_slice_window)        
        output_data["mert"] = mert_feature[start_index:start_index+self.rand_slice_window].float()
        output_data["y"] = label_feature[start_index:start_index+self.rand_slice_window].float()
        return output_data
