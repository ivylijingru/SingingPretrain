import json

import numpy as np
import torch
import torch.utils.data as Data
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
from torch import nn
import soundfile as sf
import torchaudio.transforms as T


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

        # loading our model weights
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
        # loading the corresponding preprocessor config
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0-public",trust_remote_code=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        output_data = dict()
        # load audio
        audio, sampling_rate = sf.read(self.data[idx]["vocal_path"])

        # convert to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio_array = torch.from_numpy(audio).float()

        # resample
        resample_rate = self.processor.sampling_rate
        if resample_rate != sampling_rate:
            resampler = T.Resample(sampling_rate, resample_rate)
        else:
            resampler = None
        if resampler is None:
            input_audio = audio_array
        else:
            input_audio = resampler(audio_array)
        
        # align label and audio length
        label_feature = torch.from_numpy(np.load(self.data[idx]["label_path"]))
        start_index = np.random.randint(low=0, high=label_feature.shape[0]-self.rand_slice_window)
        wave_start_index = int(start_index * resample_rate / 50) # TODO: remove hard code 50 hz
        # if we don't add this "resample_rate / 50" seems to be wrong.
        wave_end_index = int(wave_start_index + self.rand_slice_window * resample_rate / 50 + resample_rate / 50)
        wave_snippet = input_audio[wave_start_index:wave_end_index]

        # process and extract embeddings
        inputs = self.processor(wave_snippet, sampling_rate=resample_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

        # output_data["mert"] = mert_feature[start_index:start_index+self.rand_slice_window].float()
        output_data["mert"] = all_layer_hidden_states.float() # [13 layer, Time steps, 768 feature_dim]
        output_data["y"] = label_feature[start_index:start_index+self.rand_slice_window].float()

        return output_data
