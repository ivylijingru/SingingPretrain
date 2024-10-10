import os
import json
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
import torch.nn.functional as F

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


def load_audio(audio_path):
    """Load audio file and convert to mono if necessary."""
    audio, sampling_rate = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return torch.from_numpy(audio).float(), sampling_rate


def resample_audio(audio_array, original_rate, target_rate):
    """Resample audio to the target sampling rate."""
    if original_rate != target_rate:
        resampler = T.Resample(original_rate, target_rate)
        return resampler(audio_array)
    return audio_array


def get_best_checkpoint(checkpoint_dir):
    """从检查点目录中找到损失最小的检查点路径."""
    min_loss = float('inf')  # 初始化为无穷大
    min_idx = -1
    
    # 遍历检查点目录中的文件
    for ii, file_name in enumerate(os.listdir(checkpoint_dir)):
        # 提取损失值
        loss_str = file_name.split(".")[0].split("=")[-1] + file_name.split(".")[1].split("-")[0]
        
        try:
            this_loss = float(loss_str)  # 转换为浮点数
        except ValueError:
            continue  # 如果转换失败，则跳过该文件
        
        # 更新最小损失和索引
        if this_loss < min_loss:
            min_loss = this_loss
            min_idx = ii

    if min_idx == -1:
        raise ValueError("No valid checkpoint files found in the directory.")

    # 返回损失最小的检查点的完整路径
    checkpoint_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[min_idx])
    return checkpoint_path, min_idx


def process_sliced_audio(inputs, label_feature, model, mert_processor, slice_length=5):
    """
    将输入和标签切片并通过模型处理，最后生成 frame_list。

    参数:
    - inputs: 处理过的音频输入张量
    - label_feature: 标签特征，张量
    - model: 模型
    - mert_processor: 用于音频处理的处理器
    - slice_length: 每个切片的秒数 默认是5秒

    返回:
    - frame_list: 包含每个帧信息的列表
    """

    # 获取采样率并计算每个切片的帧数
    resample_rate = mert_processor.sampling_rate
    num_frames_per_slice = slice_length * resample_rate

    # 切分 inputs 和 label_feature
    total_frames = inputs.shape[-1]  # 假设输入的最后一维是帧数
    num_slices = total_frames // num_frames_per_slice
    num_tokens = num_slices * 50

    all_onset_prob = []
    all_silence_prob = []
    all_octave_logits = []
    all_pitch_logits = []

    for slice_idx in range(num_slices + 1):
        # 计算每个切片的开始和结束帧
        start_frame = slice_idx * num_frames_per_slice
        end_frame = min(start_frame + num_frames_per_slice + int(resample_rate / 50), total_frames)
        
        if end_frame <= start_frame:
            continue

        # 取出当前切片
        input_slice = inputs[..., start_frame:end_frame]  # 假设 inputs 是 (B, C, T)
        padding = (0, num_frames_per_slice + int(resample_rate / 50) - (end_frame - start_frame))
        input_slice_padded = F.pad(input_slice, padding, "constant", 0)

        # label_slice = label_feature[..., start_frame:end_frame]  # 切片 label_feature
        mert_slice = mert_processor(input_slice_padded, sampling_rate=resample_rate, return_tensors="pt").to("cuda")

        # 生成 batch，进行前向传播
        batch = {
            "inputs": mert_slice.to("cuda"),
        }
        
        # 获取模型输出
        logic_dict = model.inference_step(batch)

        # 处理输出
        onset_prob = torch.sigmoid(logic_dict["onset"][0]).cpu()
        silence_prob = torch.sigmoid(logic_dict["silence"][0]).cpu()
        octave_logits = logic_dict["octave"][0].cpu()
        pitch_logits = logic_dict["pitch"][0].cpu()

        # 将各部分结果拼接到对应的列表中
        token_nums = int((end_frame - start_frame) / resample_rate * 50)
        all_onset_prob.append(onset_prob[:token_nums])
        all_silence_prob.append(silence_prob[:token_nums])
        all_octave_logits.append(octave_logits[:token_nums])
        all_pitch_logits.append(pitch_logits[:token_nums])
        # print(token_nums, onset_prob.shape)

    # 将所有切片的输出拼接到一起
    all_onset_prob = torch.cat(all_onset_prob, dim=0)
    all_silence_prob = torch.cat(all_silence_prob, dim=0)
    all_octave_logits = torch.cat(all_octave_logits, dim=0)
    all_pitch_logits = torch.cat(all_pitch_logits, dim=0)

    half_num_frames = int(slice_length * resample_rate / 2)

    half_onset_prob = []
    half_silence_prob = []
    half_octave_logits = []
    half_pitch_logits = []

    half_num_tokens = int(slice_length * 50 / 2)
    half_onset_prob.append(all_onset_prob[:half_num_tokens])
    half_silence_prob.append(all_silence_prob[:half_num_tokens])
    half_octave_logits.append(all_octave_logits[:half_num_tokens])
    half_pitch_logits.append(all_pitch_logits[:half_num_tokens])

    for slice_idx in range(num_slices + 1):
        # 计算每个切片的开始和结束帧
        start_frame = slice_idx * num_frames_per_slice + half_num_frames
        end_frame = min(start_frame + num_frames_per_slice + int(resample_rate / 50), total_frames)
        if end_frame <= start_frame:
            continue

        # 取出当前切片
        input_slice = inputs[..., start_frame:end_frame]  # 假设 inputs 是 (B, C, T)
        padding = (0, num_frames_per_slice + int(resample_rate / 50) - (end_frame - start_frame))
        input_slice_padded = F.pad(input_slice, padding, "constant", 0)

        # label_slice = label_feature[..., start_frame:end_frame]  # 切片 label_feature
        mert_slice = mert_processor(input_slice_padded, sampling_rate=resample_rate, return_tensors="pt").to("cuda")


        # 生成 batch，进行前向传播
        batch = {
            "inputs": mert_slice.to("cuda"),
        }
        
        # 获取模型输出
        logic_dict = model.inference_step(batch)

        # 处理输出
        onset_prob = torch.sigmoid(logic_dict["onset"][0]).cpu()
        silence_prob = torch.sigmoid(logic_dict["silence"][0]).cpu()
        octave_logits = logic_dict["octave"][0].cpu()
        pitch_logits = logic_dict["pitch"][0].cpu()

        token_nums = int((end_frame - start_frame) / resample_rate * 50)
        # 将各部分结果拼接到对应的列表中
        half_onset_prob.append(onset_prob[:token_nums])
        half_silence_prob.append(silence_prob[:token_nums])
        half_octave_logits.append(octave_logits[:token_nums])
        half_pitch_logits.append(pitch_logits[:token_nums])
        # print(token_nums)

    # 将所有切片的输出拼接到一起
    half_onset_prob = torch.cat(half_onset_prob, dim=0)
    half_silence_prob = torch.cat(half_silence_prob, dim=0)
    half_octave_logits = torch.cat(half_octave_logits, dim=0)
    half_pitch_logits = torch.cat(half_pitch_logits, dim=0)

    all_onset_prob = (all_onset_prob + half_onset_prob) / 2.0
    all_silence_prob = (all_silence_prob + half_silence_prob) / 2.0
    all_octave_logits = (all_octave_logits + half_octave_logits) / 2.0
    all_pitch_logits = (all_pitch_logits + half_pitch_logits) / 2.0

    # 生成最终的 frame_list
    frame_list = [
        (
            all_onset_prob[fid],
            all_silence_prob[fid],
            torch.argmax(all_octave_logits[fid]).item(),
            torch.argmax(all_pitch_logits[fid]).item(),
        )
        for fid in range(all_onset_prob.shape[0])
    ]

    return frame_list


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
    checkpoint_path, min_idx = get_best_checkpoint(checkpoint_dir)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.to("cuda")
    model.eval()

    results = dict()

    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0-public",trust_remote_code=True)

    for idx in range(len(test_data)):
        with torch.no_grad():
            audio_array, sampling_rate = load_audio(test_data[idx]["vocal_path"])
            resample_rate = mert_processor.sampling_rate
            input_audio = resample_audio(audio_array, sampling_rate, resample_rate)
            # process and extract embeddings
            label_feature = torch.from_numpy(np.load(test_data[idx]["label_path"]))
            frame_list = process_sliced_audio(input_audio, label_feature, model, mert_processor)
            results[test_data[idx]["clip_id"]] = parse_frame_info(frame_list)

    # TODO: change naming strategy
    with open("evaluate_res_{}.json".format(os.listdir(checkpoint_dir)[min_idx]), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    import fire

    fire.Fire(test)
