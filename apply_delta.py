from typing import Optional

import argparse
import torch
import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_diff(path_raw, path_tuned, path_diff, device="cpu"):
    model_diff = AutoModelForCausalLM.from_pretrained(
        path_diff,
        device_map={"": torch.device(device)},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model_raw = AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    tokenizer_diff = AutoTokenizer.from_pretrained(path_diff)
    print('Finish loading tokenizer_diff')

    state_dict_diff = model_diff.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_diff):
        if (state_dict_raw[key].size() != state_dict_diff[key].size()):
            delta = state_dict_diff[key].size(0) - state_dict_raw[key].size(0)
            state_dict_raw[key] = torch.cat((state_dict_raw[key], torch.zeros((delta, state_dict_raw[key].size(1)), dtype=torch.bfloat16)))
            print(key)
            print(state_dict_raw[key].size(), state_dict_diff[key].size())
        state_dict_diff[key].add_(state_dict_raw[key])

    model_diff.save_pretrained(path_tuned)
    tokenizer_diff.save_pretrained(path_tuned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--tuned-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    args = parser.parse_args()

    apply_diff(args.base_model_path, args.tuned_model_path, args.delta_path)
    