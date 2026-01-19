
1. 用于在acclerate框架中，sh启动时，选择ddp还是deepspeed
./default_config.yaml
./deepspeed.yaml


2. 推理:
模型权重HuggingFace: wonderfuluuuuuuuuuuu/dLLM-ASR

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