
1. 用于在acclerate框架中，sh启动时，选择ddp还是deepspeed
./default_config.yaml
./deepspeed.yaml


训练的config
./vsr_8B.yaml
主要需要查看：
project_name:
exp_dir:
lr
weight_decay
gradient_accumulation_steps
warmup_steps
total_steps
save_interval
log_interval
batch_size
num_workers
prefetch_factor

注意：
dataset下面已被废弃不用，直接由train.sh中input_training_jsonl参数输入jsonl



2. 
asr 推理，输入格式的区别：jsonl或者是手动diy
./test_speech_chat_jsonl.py
./test_speech_chat.py

3. 
启动脚本
./train_offline_8B_0905.sh


用于平台的多机多卡
./multinode-2node.sh

用于平台的训练环境初始化
./training_environment.sh


4. 
dataset: ./twj_dataset_offline.py
model: ./merge_model.py
train: ./train_offline_8B.py