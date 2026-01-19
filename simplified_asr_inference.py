"""
A simplified script for Automatic Speech Recognition (ASR) inference.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
import os, time, librosa, re
import argparse
from peft import LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download
import os

# Assuming these utility functions are defined elsewhere and accessible.
# If not, their definitions need to be provided or replaced.
from twj_utils import load_audio, log_mel_spectrogram, compute_token_num
from merge_model import SpeechWrapper
from dinfer.model.modeling_llada_fastdllm import LLaDAModelLM as LLaDAModelLM_fastdllm

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    threshold: float = None,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)
    
    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = transfer_index | force_mask
        transfer_index = transfer_index & mask_index
        return x0, transfer_index
    
    raise ValueError("Threshold must be provided.")


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, input_embeddings=None, 
             threshold=0.9, text_tokenizer=None, early_stop=False):

    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''

    eos_id = text_tokenizer.pad_token_id

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks = gen_length // block_length
    steps = steps

    wte = model.llm.base_model.model.model.transformer.wte
    full_input_embeddings = wte(x)
    full_input_embeddings[:, :prompt.shape[1]] = input_embeddings

    nfe = 0
    for num_block in range(num_blocks):
        # block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        # # num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            # print('nfe  ', nfe)
            mask_index = (x == mask_id)
            logits = model.llm(
                inputs_embeds=full_input_embeddings,
            ).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, None, threshold)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if transfer_index.sum() > 0:
                new_embeddings = wte(x[transfer_index])
                full_input_embeddings[transfer_index] = new_embeddings

            
            if nfe== 1:
                gen_part = x[0, prompt.shape[1]:]
                
                # 查找生成部分中第一次出现 EOS 的位置
                # (gen_part == eos_token_id) 返回布尔掩码，nonzero 找到索引
                eos_indices = (gen_part == eos_id).nonzero(as_tuple=True)[0]

                if len(eos_indices) > 0:
                    # 找到第一个 EOS 的相对索引
                    first_eos_idx = eos_indices[0].item()
                    
                    # 计算新的总长度：Prompt长度 + EOS及其之前的长度
                    # +1 是为了保留这个 EOS token，表示句子结束
                    new_length = prompt.shape[1] + first_eos_idx
                    
                    # 如果新长度比当前长度短，说明可以剪枝
                    if new_length < x.shape[1]:
                        # print(f"Pruning: Reducing length from {x.shape[1]} to {new_length}")
                        
                        # 截断 token tensor
                        x = x[:, :new_length]
                        
                        # 截断 embedding tensor (这是节省计算量的关键)
                        full_input_embeddings = full_input_embeddings[:, :new_length, :]

            if early_stop:
                pos = torch.arange(x.shape[1], device = x.device).unsqueeze(0)
                eos_mask = (x == eos_id)
                first_eos = torch.where(eos_mask, pos, x.shape[1]).amin(dim=1)
                after_first = pos > first_eos.unsqueeze(1)

                x[after_first] = eos_id

            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
        
        if early_stop and  (x == eos_id).any():
            return x, nfe

    return x, nfe


def main():
    import os

    parser = argparse.ArgumentParser(description="Run ASR on a single audio file.")
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the input audio file.')
    # parser.add_argument('--model_ckpt', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for inference.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = 'cuda:0'

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("wonderfuluuuuuuuuuuu/dLLM-ASR", trust_remote_code=True)

    forzero_id = tokenizer.convert_tokens_to_ids("<|forzero|>")

    model_path = 'GSAI-ML/LLaDA-8B-Instruct'
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    config.train_max_sequence_length = config.max_sequence_length
    model = LLaDAModelLM_fastdllm.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model = SpeechWrapper(llm=model)

    # 下载文件到本地（默认下载到缓存目录，也可指定local_dir）
    local_file_path = hf_hub_download(
        repo_id="wonderfuluuuuuuuuuuu/dLLM-ASR",
        filename="epoch_5_step_450000.pt",  # 模型权重文件
        repo_type="model",
        local_dir="./huggingface_files"  # 自定义本地保存目录，不存在会自动创建
    )
    print(local_file_path)
    state_dict = torch.load(local_file_path, map_location='cpu')
    m, u = model.load_state_dict(state_dict)
    print(f'GPU {args.gpu_id} missing keys: {m}')
    print(f'GPU {args.gpu_id} unexpected keys: {u}')
    model = model.to(torch.bfloat16).to(device).eval()

    # Prepare audio
    sample_rate = 16000
    audio = load_audio(file_path=args.audio_path, target_rate=sample_rate)
    mel = log_mel_spectrogram(audio=audio, n_mels=128, padding=0, device=None)
    speech_frames = compute_token_num(mel.shape[1])
    hidden_states = mel.to(device).unsqueeze(0)

    # Prepare prompt
    forzero_str = "<|forzero|>"
    forzero_str_withlength = f"{forzero_str * speech_frames}"
    wrapped = [{"role": "user", "content": forzero_str_withlength + "Transcribe this audio into text."}]
    user_input = tokenizer.apply_chat_template(wrapped, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(user_input, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)

    noisy_input_embeddings = model.llm.base_model.model.model.transformer.wte(input_ids)
    speech_lens = torch.LongTensor([hidden_states.shape[2]]).to(device)
    noisy_input_embeddings = model.prepare_speech_inputs(
        input_ids=input_ids,
        noisy_input_embeddings=noisy_input_embeddings,
        wavs=hidden_states,
        wav_lens=speech_lens,
        forzero_id=forzero_id
    )

    # Generate text
    # A reasonable guess for gen_length, can be adjusted.
    gen_length = 256 
    out = generate(
        model, input_ids,
        gen_length=gen_length,
        temperature=0.0,
        threshold=0.9,
        remasking='low_confidence',
        input_embeddings=noisy_input_embeddings,
        text_tokenizer=tokenizer
    )

    predicted_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    print("\n--- ASR Result ---")
    print(f"Audio file: {args.audio_path}")
    print(f"Predicted text: {predicted_text}")
    print("------------------\n")

if __name__ == '__main__':
    main()