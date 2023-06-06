"""
Apply the delta weights on top of a base model.

Usage:
python3 apply_delta.py --base ~/model_weights/llama-13b --target ~/model_weights/yulan-13b --delta ~/model_weights/yulan-13b-delta
"""
import argparse
import gc
import glob
import json
import os
import shutil
import tempfile

from huggingface_hub import snapshot_download
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def apply_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the delta weights from {delta_path}")
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)
    delta = AutoModelForCausalLM.from_pretrained(
        delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    if base.model.embed_tokens.weight.size(0) + 1 == delta.model.embed_tokens.weight.size(0):
        print(base.model.embed_tokens.weight)
        base.resize_token_embeddings(len(delta_tokenizer))
        print(base.model.embed_tokens.weight)
        base.model.embed_tokens.weight.data[-1, :] = 0
        print(base.model.embed_tokens.weight)

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
