import json

import numpy as np
import torch
import torch.utils.data as Data


class TranscriptionDataset(Data.Dataset):
    def __init__(
        self,
        manifest_path,
    ) -> None:
        super().__init__()

        with open(manifest_path) as f:
            self.data = [json.loads(line) for line in f]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        output_data = dict()
        output_data["mert"] = torch.from_numpy(np.load(data["mert_path"]))
        output_data["y"] = torch.from_numpy(np.load(data["label_path"]))

        return output_data
