"""
Generating the time sequence; 
Command line: python evaluate.py ../MIR-ST500_20210206/MIR-ST500_corrected.json ../../SingingPretrain/evaluate_res_update.json 0.05
results:

        Precision Recall F1-score
COnPoff 0.246968 0.266422 0.253591
COnP    0.431568 0.462061 0.440878
COn     0.653907 0.687490 0.661262

"""
import json
import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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

    model = model.load_from_checkpoint("work_dir_mert/log/version_26/checkpoints/epoch=159-step=1120.ckpt")
    model.eval()

    results = dict()

    for idx in range(len(test_data)):
        with torch.no_grad():
            mert_feature = torch.from_numpy(np.load(test_data[idx]["mert_path"]))
            # mert_feature = torch.squeeze(mert_feature, 0)
            label_feature = torch.from_numpy(np.load(test_data[idx]["label_path"]))
            batch = dict()
            batch["mert"] = mert_feature
            batch["y"] = label_feature.unsqueeze(0)
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

    with open("evaluate_res.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    config = "config/svt_mert_debug.json"
    test(config)