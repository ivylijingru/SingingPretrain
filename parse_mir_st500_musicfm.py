import json
import os
from tqdm import tqdm
import soundfile as sf

import warnings
import librosa
import numpy as np
import pandas as pd


def parse_mirst500(data_split, output_dir):
    vocal_dir = "../singing_transcription_ICASSP2021"
    label_dir = "MusicFM-label"

    data_list = []
    for the_dir in tqdm(os.listdir(os.path.join(label_dir, data_split))):
        clip_id = int(the_dir)
        vocal_path = os.path.join(vocal_dir, data_split, the_dir, "Vocals.wav")
        label_path = os.path.join(label_dir, data_split, the_dir, "labels.npy")

        data = dict(
            clip_id=clip_id,
            vocal_path=vocal_path,
            label_path=label_path,
        )

        data_list.append(data)
    
    with open(os.path.join(output_dir, "mirst500-musicfm-{}.json".format(data_split)), "w") as f:
        for data in data_list:
            json.dump(data, f)
            f.write('\n')
            f.flush()


if __name__ == '__main__':
    parse_mirst500("train", "data_json_musicfm")
    parse_mirst500("test", "data_json_musicfm")