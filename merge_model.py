from logging import config
import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from typing import Optional
import sys
import transformers
import wandb
import numpy as np
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
import math


import torch.nn as nn

class SpeechWrapper(nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        n_state=1024
        kernel_size=4
        stride=2
        padding = (kernel_size - stride) // 2
        # 在时间维度上卷积做下采样

        self.speech_proj = nn.Sequential(
            nn.Linear(3584, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 4096)
        )
        
        self.speech_encoder = AutoModelForCausalLM.from_pretrained("/mnt/bn/twj-data-multimodal2/workspace/Step-Audio-2-mini", trust_remote_code=True, torch_dtype=torch.bfloat16)
        del self.speech_encoder.lm_head
        del self.speech_encoder.model
        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        self.speech_encoder.eval()
        

# 用于inference input preparation
    def prepare_speech_inputs(
        self,
        input_ids=None,
        noisy_input_embeddings=None,
        wavs=None,
        wav_lens=None,
        forzero_id=None,
    ):
        hidden_states = noisy_input_embeddings
        # if self.bf16:
        wavs = wavs.bfloat16()
        out, feat_lens = self.speech_encoder.encoder(wavs, wav_lens)
        out = self.speech_encoder.adapter(out)


        feat_lens = (feat_lens - 1) // 2 + 1 # 下采样后长度

        # out = self.speech_proj(out.to(noisy_input_embeddings.dtype))
        out = self.speech_proj(out)

        # Vectorized replacement of placeholders with speech features.
        # This is much faster than a for-loop and corrects a potential off-by-one error.
        placeholder_mask = (input_ids == forzero_id)
        placeholder_lens = placeholder_mask.sum(dim=1)

        # Optional: A check for length mismatch.
        if not torch.equal(placeholder_lens, feat_lens):
            print(f"Warning: Mismatch between number of placeholders:{placeholder_lens} and speech feature lengths found:{feat_lens}.")


        # Create a mask for the valid speech features in the 'out' tensor.
        out_mask = torch.arange(out.shape[1], device=out.device)[None, :] < feat_lens[:, None]

        # Perform the replacement if total lengths match.
        if placeholder_mask.sum() == out_mask.sum():
            noisy_input_embeddings[placeholder_mask] = out[out_mask]
        else:
            # Fallback to a corrected loop if total counts mismatch.
            print("Warning: Total placeholder count and feature count differ. Falling back to a corrected loop.")
            for i in range(input_ids.shape[0]):
                if placeholder_lens[i] > 0:
                    indices_to_update = placeholder_mask[i].nonzero(as_tuple=True)[0]
                    len_to_copy = min(len(indices_to_update), feat_lens[i])
                    noisy_input_embeddings[i, indices_to_update[:len_to_copy], :] = out[i, :len_to_copy, :]

        return hidden_states

        
    def forward(self, 
                noisy_input_embeddings, input_ids, input_ids_mask, forzero_id, 
                speech_encoder_out=None, speech=None, speech_lens=None, 
                ):
        batch_size_actual = noisy_input_embeddings.size(0)
        avg_input_length = [1]

        # speech_mask = torch.zeros_like(input_ids_mask, dtype=torch.float32) # input_ids_mask是padding mask
        noisy_input_embeddings = self.prepare_speech_inputs(
                                        input_ids=input_ids,
                                        noisy_input_embeddings=noisy_input_embeddings,
                                        wavs=speech.transpose(1,2),
                                        wav_lens=speech_lens,
                                        forzero_id=forzero_id    #     "id": 151688, "content": "<audio_start>",
                                        )

        model_output = self.llm(
            inputs_embeds=noisy_input_embeddings,
            attention_mask=input_ids_mask
        )
        return model_output, avg_input_length

    def count_parameters(self):
        """
        计算并打印模型的参数统计信息
        
        参数:
            model: 可选，指定要统计的模型，默认使用当前实例
        """
        # 确定要统计的模型
        target_model = self
        
        # 统计可训练参数和总参数
        trainable_params = 0
        all_params = 0
        
        for param in target_model.parameters():
            num_params = param.numel()  # 获取参数数量
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        # 计算可训练参数百分比
        trainable_percent = (trainable_params / all_params) * 100 if all_params > 0 else 0
        
        # 格式化输出（添加千位分隔符）
        info = f"trainable params: {trainable_params:,} || "
        info += f"all params: {all_params:,} || "
        info += f"trainable%: {trainable_percent:.4f}"
        print(
            info
        )

        return info
