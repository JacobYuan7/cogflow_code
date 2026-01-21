#!/usr/bin/env bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS

export MODEL_DIR=MODEL_PATH

# input.jsonl 
export INPUT_JSONL=DATA_PATH/infer.jsonl
export OUTPUT_JSONL=DATA_PATH/infer_out_visual_gate.jsonl

python3 infer_visual_gate.py \
  --model_path $MODEL_DIR \
  --input_jsonl $INPUT_JSONL \
  --output_jsonl $OUTPUT_JSONL \
  \
  --tp 1 \
  --gpu_mem_util 0.7 \
  --max_model_len 2048 \
  --dtype bfloat16 \
  --trust_remote_code \
  \
  --tau 0.75 \
  --max_trials 3 \
  --perception_max_new_tokens 128 \
  --reasoning_max_new_tokens 256 \
  --stop_strings '["<REASONING>"]' \
  --continue_prefix "<REASONING>\n" \
  \
  --temperature 0.7 \
  --top_p 0.95 \
  --seed 1234 \
  \
  --clip_name qihoo360/fg-clip-large
