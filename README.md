# CogFlow RL (VERL + vLLM) 

This repository supports **PPO-like / VGPO reinforcement learning** training  for text / multimodal models.  

## 1. Repository Structure

The structure below matches the current repository layout:

```
├── .vscode/
├── data/ # Data directory (train/val/infer)
├── docker/ # Docker environments
├── docs/ # Documentation
├── outputs/ # Training / inference outputs (logs, ckpts, jsonl)
├── recipe/ # Training recipe configs (if any)
├── reward/
│ └── vgpo_reward.py # ✅ Custom reward function for VERL training
├── scripts/ # Helper scripts (optional)
├── src/
│ ├── cogflow_process_data.py # ✅ Data preprocessing script
│ ├── infer.py # ✅ Inference entry (with VSR gating)
│ ├── infer.sh # ✅ Inference launch script
│ └── train_vgpo.sh # ✅ VGPO training launch script
├── tests/
├── verl/ # ✅ VERL source code (your local rollout modifications are here)
├── verl.egg-info/
├── pyproject.toml
├── requirements.txt
├── requirements_sglang.txt
├── requirements-npu.txt
├── setup.py
└── README.md
```


---

## 2. Environment & Installation

### 2.1 Python & CUDA

Recommended:
- Python >= 3.10
- CUDA >= 11.8 (depends on your torch/vLLM versions)
- Multi-GPU training: 8 GPUs (adjustable)

### 2.2 Install Dependencies

Install base dependencies:

```bash
pip install -r requirements.txt
```

# 3. Data Preparation

Preprocessing script:
```bash
src/cogflow_process_data.py
```


Example usage:
```bash
python src/cogflow_process_data.py \
  --input data/raw \
  --output data/processed
```

## 3.1 Training Data Format (VERL)

Training uses parquet by default:

- data/train.parquet

- data/val.parquet

Inference often uses jsonl (or any custom format):

- data/infer.jsonl

# 4. RL Training (VGPO / PPO-like)

Training entry script:
```bash

bash src/train_vgpo.sh
```



# 5. Custom Reward

Custom reward file:

```bash
reward/vgpo_reward.py
```

It is passed to training via:

`custom_reward_function.path=reward/vgpo_reward.py`

`custom_reward_function.name=compute_score
`

Inside `compute_score()` you can implement:

- rule-based reward

- model-based reward (e.g., scoring with IntlzR reward model)

- tool-based reward (e.g., sandbox test cases)

# 6. Visual-Gated Inference

Inference entry points:

`src/infer.py`

`src/infer.sh`





# 7. Run Inference

Run:
```bash
bash src/infer.sh
```

Outputs are saved to:
```bash
outputs/infer/<experiment_name>/*.jsonl
```

# 9. Swift Training (SFT + IntlzR Reward Model)

⚠️ Important: SFT and IntlzR Reward Model are trained using the Swift framework, NOT inside this repository.

This repo is responsible for:

- loading Swift-exported checkpoints as MODEL_DIR

- optionally calling the Swift-trained reward model during reward computation or inference scoring

Artifacts usage:

- SFT checkpoint: used as actor_rollout_ref.model.path=$MODEL_DIR

- IntlzR reward model: can be invoked inside reward/vgpo_reward.py or src/infer.py

# 10. Output & Logging
## 10.1 Training Outputs

Training outputs include:

logs: outputs/ or exp_log/

checkpoints: outputs/ckpt/ (depends on your trainer configs)




# License
```bash 
This project contains ByteDance VERL code and follows the Apache 2.0 License.
```

# Citation 
```bibtex
@article{chen2026cogflow,
  title   = {CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving},
  author  = {Chen, Shuhang and Xu, Yunqiu and Xie, Junjie and Lu, Aojun and Feng, Tao and Huang, Zeying and Zhang, Ning and Sun, Yi and Yang, Yi and Yuan, Hangjie},
  journal = {arXiv preprint arXiv:2601.01874},
  year    = {2026}
}