from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random
import concurrent.futures
import pickle
import time
import json
import soundfile as sf


def preprocess(gt_data, length, pitch_shift=0):

    new_label = []

    cur_note = 0
    cur_note_onset = gt_data[cur_note][0]
    cur_note_offset = gt_data[cur_note][1]
    cur_note_pitch = gt_data[cur_note][2] + pitch_shift

    # start from C2 (36) to B5 (83), total: 4 classes. This is a little confusing
    octave_start = 0
    octave_end = 3
    pitch_class_num = 12

    frame_size = 1 / 50 # 0.02 s; or 20 ms

    for i in range(length):
        cur_time = i * frame_size

        if abs(cur_time - cur_note_onset) <= (frame_size / 2.0):
            # First dim : onset
            # Second dim : silence
            if i == 0 or new_label[-1][0] != 1:
                my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                my_pitch_class = cur_note_pitch % pitch_class_num
                label = [1, 0, my_oct, my_pitch_class]
                new_label.append(label)
            else:
                my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                my_pitch_class = cur_note_pitch % pitch_class_num
                label = [0, 0, my_oct, my_pitch_class]
                new_label.append(label)

        elif cur_time < cur_note_onset or cur_note >= len(gt_data):
            # For the frame that doesn't belong to any note
            label = [0, 1, octave_end+1, pitch_class_num]
            new_label.append(label)

        elif abs(cur_time - cur_note_offset) <= (frame_size / 2.0):
            # For the offset frame
            my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
            my_pitch_class = cur_note_pitch % pitch_class_num
            label = [0, 1, my_oct, my_pitch_class]

            cur_note = cur_note + 1
            if cur_note < len(gt_data):
                cur_note_onset = gt_data[cur_note][0]
                cur_note_offset = gt_data[cur_note][1]
                cur_note_pitch = gt_data[cur_note][2] + pitch_shift
                if abs(cur_time - cur_note_onset)  <= (frame_size / 2.0):
                    my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                    my_pitch_class = cur_note_pitch % pitch_class_num
                    label[0] = 1
                    label[1] = 0
                    label[2] = my_oct
                    label[3] = my_pitch_class

            new_label.append(label)

        else:
            # For the voiced frame
            my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
            my_pitch_class = cur_note_pitch % pitch_class_num

            label = [0, 0, my_oct, my_pitch_class]
            new_label.append(label)

    return np.array(new_label)


if __name__ == "__main__":
    gt_path = "MIR-ST500_20210206/MIR-ST500_corrected.json"
    with open(gt_path) as json_data:
        gt = json.load(json_data)
    
    label_dir = "MusicFM-label"
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    if not os.path.exists(os.path.join(label_dir, "train")):
        os.makedirs(os.path.join(label_dir, "train"))

    if not os.path.exists(os.path.join(label_dir, "test")):
        os.makedirs(os.path.join(label_dir, "test"))

    vocal_dir = "../singing_transcription_ICASSP2021"
    train_data_dir = os.path.join(vocal_dir, "train")
    train_output_dir = os.path.join(label_dir, "train")
    for the_dir in tqdm(os.listdir(train_data_dir)):
        gt_data = gt[the_dir]
        
        # read audio and get its length
        vocal_path = os.path.join(train_data_dir, the_dir, "Vocals.wav")
        audio, sampling_rate = sf.read(vocal_path)
        feature_length = int(audio.shape[0] / sampling_rate * 50)

        # preprocess data
        answer_data = preprocess(gt_data, feature_length)
        
        # write data to disk
        cur_dir = os.path.join(train_output_dir, the_dir)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        output_path = os.path.join(cur_dir, "labels.npy")
        np.save(output_path, answer_data)

    vocal_dir = "../singing_transcription_ICASSP2021"
    test_data_dir = os.path.join(vocal_dir, "test")
    test_output_dir = os.path.join(label_dir, "test")
    for the_dir in tqdm(os.listdir(test_data_dir)):
        gt_data = gt[the_dir]
        
        # read audio and get its length
        vocal_path = os.path.join(test_data_dir, the_dir, "Vocals.wav")
        audio, sampling_rate = sf.read(vocal_path)
        feature_length = int(audio.shape[0] / sampling_rate * 50)

        # preprocess data
        answer_data = preprocess(gt_data, feature_length)
        
        # write data to disk
        cur_dir = os.path.join(test_output_dir, the_dir)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        output_path = os.path.join(cur_dir, "labels.npy")
        np.save(output_path, answer_data)
