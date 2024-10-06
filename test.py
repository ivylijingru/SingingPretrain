"""
Generating the time sequence; 
Command line: python evaluate.py ../MIR-ST500_20210206/MIR-ST500_corrected.json ../../SingingPretrain/evaluate_res_update.json 0.05
results:

         Precision Recall F1-score
COnPOff  0.257557 0.282073 0.268348
COnP     0.479153 0.527897 0.500631
COn      0.680818 0.751427 0.711923

(marble) jli3268@aurora:~/singing_transcription_ICASSP2021$ python evaluate/evaluate.py MIR-ST500_20210206/MIR-ST500_corrected.json ../SingingPretrain/evaluate_res_epoch\=27-val_loss-total\=2.873.ckpt.json 0.05
1727710026.0981421
         Precision Recall F1-score
COnPOff  0.325149 0.302426 0.312507
COnP     0.600014 0.558588 0.576929
COn      0.769386 0.715108 0.739075
gt note num: 31311.0 tr note num: 29096.0

MLP 1024
(marble) jli3268@aurora:~/singing_transcription_ICASSP2021$ python evaluate/evaluate.py MIR-ST500_20210206/MIR-ST500_corrected.json ../SingingPretrain/evaluate_res_epoch\=27-val_loss-total\=3.122.ckpt.json 0.05
1727710511.492257
         Precision Recall F1-score
COnPOff  0.306619 0.332593 0.318137
COnP     0.540829 0.591287 0.563198
COn      0.704345 0.770360 0.733627
gt note num: 31311.0 tr note num: 34250.0
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import torchaudio.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel

from data import TranscriptionDataModule
from models import SVTDownstreamModel


FRAME_LENGTH = 0.02

def parse_frame_info(frame_info, onset_thres=0.4, offset_thres=0.5):
    """Parse frame info [(onset_probs, offset_probs, pitch_class)...] into desired label format."""

    result = []
    current_onset = None
    pitch_counter = []

    last_onset = 0.0
    onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])

    local_max_size = 3
    current_time = 0.0

    onset_seq_length = len(onset_seq)

    for i in range(len(frame_info)):

        current_time = FRAME_LENGTH*i
        info = frame_info[i]

        backward_frames = i - local_max_size
        if backward_frames < 0:
            backward_frames = 0

        forward_frames = i + local_max_size + 1
        if forward_frames > onset_seq_length - 1:
            forward_frames = onset_seq_length - 1

        # local max and more than threshold
        if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):

            if current_onset is None:
                current_onset = current_time
                last_onset = info[0] - onset_thres
            else:
                if len(pitch_counter) > 0:
                    result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])

                current_onset = current_time
                last_onset = info[0] - onset_thres
                pitch_counter = []

        elif info[1] >= offset_thres:  # If is offset
            if current_onset is not None:
                if len(pitch_counter) > 0:
                    result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                current_onset = None

                pitch_counter = []

        # If current_onset exist, add count for the pitch
        if current_onset is not None:
            final_pitch = int(info[2]* 12 + info[3])
            if info[2] != 4 and info[3] != 12:
            # if final_pitch != 60:
                pitch_counter.append(final_pitch)

    if current_onset is not None:
        if len(pitch_counter) > 0:
            result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
        current_onset = None

    return result


def test(config):
    with open(config) as f:
        config = json.load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    model = SVTDownstreamModel(model_cfg)

    test_manifest_path = data_cfg["test_manifest_path"]

    with open(test_manifest_path) as f:
        test_data = [json.loads(line) for line in f]

    checkpoint_dir = config["trainer"]["checkpoint"]["dirpath"]
    min_loss = 10000
    min_idx = -1
    for ii, file_name in enumerate(os.listdir(checkpoint_dir)):
        this_loss = file_name.split(".")[0].split("=")[-1] + file_name.split(".")[1].split("-")[0]
        this_loss = float(this_loss)
        if this_loss < min_loss:
            min_loss = this_loss
            min_idx = ii
    checkpoint_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[min_idx])

    model = model.load_from_checkpoint(checkpoint_path)
    model.to("cuda")
    model.eval()

    results = dict()

    # loading our model weights
    # mert_model = AutoModel.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
    # mert_model.eval()
    # loading the corresponding preprocessor config
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0-public",trust_remote_code=True)
    # mert_model.to("cuda")

    for idx in range(len(test_data)):
        with torch.no_grad():

            # load audio
            audio, sampling_rate = sf.read(test_data[idx]["vocal_path"])

            # convert to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio_array = torch.from_numpy(audio).float()

            # resample
            resample_rate = mert_processor.sampling_rate
            if resample_rate != sampling_rate:
                resampler = T.Resample(sampling_rate, resample_rate)
            else:
                resampler = None
            if resampler is None:
                input_audio = audio_array
            else:
                input_audio = resampler(audio_array)
            
            # process and extract embeddings
            inputs = mert_processor(input_audio, sampling_rate=resample_rate, return_tensors="pt").to("cuda")
            # with torch.no_grad():
            #     outputs = mert_model(**inputs, output_hidden_states=True)
            #     all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

            # mert_feature = all_layer_hidden_states.float().unsqueeze(0)
            # # mert_feature = torch.squeeze(mert_feature, 0)
            label_feature = torch.from_numpy(np.load(test_data[idx]["label_path"]))

            batch = dict()
            batch["inputs"] = inputs
            # batch["mert"] = mert_feature.to("cuda")
            batch["y"] = label_feature.unsqueeze(0).to("cuda")
            _, logic_dict = model.common_step(batch)

            onset_prob = torch.sigmoid(logic_dict["onset"][0]).cpu()
            silence_prob = torch.sigmoid(logic_dict["silence"][0]).cpu()
            octave_logits = logic_dict["octave"][0].cpu()
            pitch_logits = logic_dict["pitch"][0].cpu()

            frame_list = []
            for fid in range(onset_prob.shape[0]):
                frame_info = (
                    onset_prob[fid], 
                    silence_prob[fid], 
                    torch.argmax(octave_logits[fid]).item(),
                    torch.argmax(pitch_logits[fid]).item(),
                )
                # print(frame_info)
                frame_list.append(frame_info)

            results[test_data[idx]["clip_id"]] = parse_frame_info(frame_list)

    # TODO: change naming strategy
    with open("evaluate_res_{}.json".format(os.listdir(checkpoint_dir)[min_idx]), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    import fire

    fire.Fire(test)
