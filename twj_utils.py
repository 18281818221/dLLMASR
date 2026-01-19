import pyarrow.parquet as pq
import io
import pandas as pd
from torch import nn
import torch
import json


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


import librosa
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import librosa
from typing import List


def _mel_filters(n_mels: int) -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram."""
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    if n_mels == 128:
        return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128))
    else:
        return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=80))

def load_audio(file_path, target_rate=16000, max_length=None):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    If max_length is provided, truncate the audio to that length
    """
    # waveform, sample_rate = torchaudio.load(file_path)
    # if sample_rate != target_rate:
    #     waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)(waveform)
    # audio = waveform[0]  # get the first channel
    audio, _ = librosa.load(file_path, sr=target_rate, mono=True)
    audio = torch.from_numpy(audio)

    # Truncate audio if it exceeds max_length
    if max_length is not None and audio.shape[0] > max_length:
        audio = audio[:max_length]

    

    return audio

def log_mel_spectrogram(audio, n_mels=128, padding=0, device=None):
    """
    Compute the log-Mel spectrogram with specific padding for StepAudio
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

def compute_token_num(max_feature_len):
    # First, audio goes through encoder:
    # 1. conv1: kernel=3, stride=1, padding=1 -> size unchanged
    # 2. conv2: kernel=3, stride=2, padding=1 -> size/2
    # 3. avg_pooler: kernel=2, stride=2 -> size/2
    max_feature_len = max_feature_len  # remove padding
    encoder_output_dim = (max_feature_len + 1) // 2 // 2  # after conv2 and avg_pooler

    # Then through adaptor (parameters from config file):
    padding = 1
    kernel_size = 3  # from config: audio_encoder_config.kernel_size
    stride = 2      # from config: audio_encoder_config.adapter_stride
    adapter_output_dim = (encoder_output_dim + 2 * padding - kernel_size) // stride + 1
    return adapter_output_dim

def padding_mels(data: List[torch.Tensor]):
    """ Padding the data into batch data

    Parameters
    ----------
        data: List[Tensor], shape of Tensor (128, T)

    Returns:
    -------
        feats, feats lengths
    """
    sample = data
    assert isinstance(sample, list)
    feats_lengths = torch.tensor([s.size(1)-2 for s in sample],
                                dtype=torch.int32)
    feats = [s.t() for s in sample]
    padded_feats = pad_sequence(feats,
                                batch_first=True,
                                padding_value=0)

    return padded_feats.transpose(1, 2), feats_lengths

def stft_loss(reconstructed: Tensor, target: Tensor, n_fft: int = 1024, hop_length: int = 256) -> Tensor:
    """
    计算STFT频谱损失，衡量频域上的差异
    
    Args:
        reconstructed: 重构的音频张量，形状为 (batch_size, channels, length)
        target: 目标音频张量，形状同上
        n_fft: STFT的FFT窗口大小
        hop_length: STFT的步长
        
    Returns:
        标量的STFT损失
    """
    # 计算STFT（返回复数谱）
    recon_stft = torch.stft(
        reconstructed.squeeze(1),  # 去掉通道维度，形状变为 (batch, length)
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    target_stft = torch.stft(
        target.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    
    # 计算幅度谱的L1损失（对人耳更敏感）
    recon_mag = torch.abs(recon_stft)
    target_mag = torch.abs(target_stft)
    return F.l1_loss(recon_mag, target_mag)

def audio_reconstruction_loss(reconstructed: Tensor, target: Tensor, mask: Tensor = None, alpha: float = 0.5) -> tuple[Tensor, Tensor, Tensor]:
    """
    音频重构损失函数，结合时域MSE和频域STFT损失，返回三个损失值
    
    Args:
        reconstructed: 重构的音频张量，形状为 (batch_size, channels, length)
        target: 目标音频张量，形状同上
        mask: 音频掩码，形状同上（用于屏蔽无效区域，如padding部分）
        alpha: STFT损失的权重（0~1之间）
        
    Returns:
        tuple: (总损失, MSE损失, STFT损失) 均为标量张量
    """
    
    # 应用掩码（如果提供）
    if mask is not None:
        reconstructed = reconstructed * mask
        target = target * mask
    
    # 时域MSE损失（捕捉波形整体相似性）
    mse_loss = F.mse_loss(reconstructed, target)
    
    # 频域STFT损失（捕捉频谱细节）
    stft_loss_val = stft_loss(reconstructed, target)
    
    # 总损失 = (1-alpha)*MSE + alpha*STFT
    total_loss = (1 - alpha) * mse_loss + alpha * stft_loss_val
    
    # 返回三个损失值
    return total_loss, mse_loss, stft_loss_val

# # 示例用法
# if __name__ == "__main__":
#     # 模拟输入：batch_size=2, 1个声道, 音频长度=16000（1秒@16kHz）
#     batch = {
#         "audios": torch.randn(2, 1, 16000),  # 目标音频
#         "audio_mask": torch.ones(2, 1, 16000)  # 掩码（全为1表示无padding）
#     }
#     reconstructed_audio = torch.randn(2, 1, 16000)  # 重构的音频
    
#     # 计算损失
#     total_loss, mse_loss, stft_loss_val = audio_reconstruction_loss(
#         reconstructed=reconstructed_audio,
#         target=batch["audios"],
#         mask=batch["audio_mask"],
#         alpha=0.3  # 给STFT损失30%的权重
#     )
    
#     print(f"总重构损失: {total_loss.item()}")
#     print(f"时域MSE损失: {mse_loss.item()}")
#     print(f"频域STFT损失: {stft_loss_val.item()}")
    
    
def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))

def read_jsonl( jsonl_list ):
    data = []
    for jsonl in jsonl_list:
        print(f'read jsonl {jsonl}')
        with open(jsonl, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data

def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl

def get_mean_stdev_from_stableaudio2_latents(mean_scale_latent):
    # mean_scale_latent torch.Size([2, 261, 128])
    mean, scale = mean_scale_latent.chunk(2, dim=1)
    stdev = nn.functional.softplus(scale) + 1e-4
    return mean, stdev

def read_parquet(parquet_paths):

    # 定义多个路径（支持文件夹、文件列表或通配符）
    # parquet_paths = [
    #     "/mnt/bn/twj-data-multimodal2/libritts_r/data/test.clean/",
    #     "/mnt/bn/twj-data-multimodal2/libritts_r/data/train.other.500",
    #     "/mnt/bn/twj-data-multimodal2/libritts_r/data/train.clean.360",
    #     "/mnt/bn/twj-data-multimodal2/libritts_r/data/train.clean.100"
    #     # "/path/to/folder_with_parquets/"  # 文件夹下的所有 Parquet 文件
    # ]
    total_df = pd.DataFrame()
    for dir_name in parquet_paths:
        dataset = pq.ParquetDataset(dir_name)   
        table = dataset.read()
        df = table.to_pandas()
        total_df = pd.concat([total_df, df], axis=0)

    return total_df
        # audio_bytes = df.iloc[0]['audio']['bytes']  # 提取二进制数据
        # text_normalized = df.iloc[0]['text_normalized']
        # data_id = df.iloc[0]['id']
        # # 将二进制数据转换为音频流（类似文件对象）
        # audio_io = io.BytesIO(audio_bytes)

        # # 方法2：用librosa读取（更适合后续音频特征处理）
        # # 重置流指针到开头
        # audio_io.seek(0)  # 重置流指针到开头
        # waveform, sample_rate = librosa.load(audio_io, sr=None)  # sr=None保留原始采样率
        # print(waveform.shape)
        # print(sample_rate)
