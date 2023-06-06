import argparse
import torch
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers import LlamaTokenizer, LlamaTokenizerFast
import warnings
warnings.filterwarnings("ignore")


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "[UNK]"


def load(args):
    model_path = args.model_path
    if args.load_in_8bit:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)
    else:
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(model, model_path, device_map="auto", no_split_module_classes=model._no_split_modules, dtype=torch.float16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    if tokenizer.bos_token == '' or tokenizer.eos_token == '' or tokenizer.unk_token == '':
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model.config.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    return model, tokenizer


PROMPT = (
    "The following is a conversation between a human and an AI assistant namely YuLan, developed by GSAI, Renmin University of China. "
    "The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "[|Human|]:{instruction}\n[|AI|]:"
)


class RemoveEmptyCharLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(self.tokenizer, (LlamaTokenizerFast, LlamaTokenizer)):
            scores[:, 30166] = float('-inf')  # remove \u200b
        return scores


@torch.inference_mode(mode=True)
def generate_response(model, tokenizer, prompt, input_text, max_length, **kwargs):
    if isinstance(input_text, str):
        input_text = [input_text]
    input_text = [prompt.format_map(dict(instruction=in_text)) for in_text in input_text]
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_attention_mask=True
    )
    kwargs.update({'max_length': max_length})

    device = next(iter(model.parameters())).device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    if input_ids.size(-1) > 2048:
        kwargs.update({'use_cache': False, 'max_new_tokens': 256})

    processors = LogitsProcessorList()
    processors.append(RemoveEmptyCharLogitsProcessor(tokenizer))
    outputs = model.generate(input_ids, attention_mask=attention_mask, logits_processor=processors, **kwargs)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    new_input_text = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
    del input_ids
    del attention_mask
    response_text = [
        out_txt[len(in_txt):].strip()
        for in_txt, out_txt in zip(new_input_text, output_text)
    ]
    return response_text


def main(args):
    model, tokenizer = load(args)
    while True:
        input_text = input('[|Human|]:')
        kwargs = {
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 64,
            "min_new_tokens": 1,
        }
        output_content = generate_response(model, tokenizer, PROMPT, input_text, max_length=2048, **kwargs)
        print(f'[|AI|]:{output_content[0]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="", help="path to model")
    parser.add_argument("--load_in_8bit", default=False, action="store_true")

    args, _ = parser.parse_known_args()
    main(args)
