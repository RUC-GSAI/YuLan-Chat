<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Chat/blob/main/assets/YuLan-logo.jpg" width="400px">
<h1>YuLan-Chat: An Open-Source Bilingual Chatbot</h1>
<a href="https://github.com/RUC-GSAI/YuLan-Chat/blob/main/LICENSE"><img src="https://img.shields.io/badge/MIT-License-blue" alt="license"></a>
<a><img src="https://img.shields.io/github/stars/RUC-GSAI/YuLan-Chat"></a>
</div>

YuLan-Chat models are chat-based large language models, which are developed by the researchers in GSAI, Renmin University of China (YuLan, which represents Yulan Magnolia, is the campus flower of Renmin University of China). The newest version is developed by pretraining from scratch, and supervised fine-tuning via curriculum learning with high-quality English and Chinese instructions and human preference data. The model has the following technical characteristics:
- Owing to large-scale pre-training on high-quality Chinese-English bilingual data, the language ability of the model has been improved.
- Owing to the curriculum learning strategy for human alignment, the helpfulness, honesty, and harmlessness of our model have been enhanced.
- To well support Chinese longer inputs and outputs, we expand the vocabulary with Chinese words and the maximum input length. It can support 8k context now.

> YuLan-Chat系列模型是中国人民大学高瓴人工智能学院师生共同开发的支持聊天的大语言模型（名字"玉兰"取自中国人民大学校花）。最新版本从头完成了整个预训练过程，并采用课程学习技术基于中英文双语数据进行有监督微调，包括高质量指令和人类偏好数据。该版模型具有如下技术特点：
> - 由于在大规模中英双语数据上进行了继续预训练，模型的语言能力得到提高；
> - 由于采用了课程学习方法进行人类对齐训练，模型在真实场景下的有用性、诚实性与无害性得到了增强；
> - 为了更好的支持中文和更长的输入输出，模型的词表及长度得到了扩充，目前可支持8k上下文。


## News

* **\[Apr. 12, 2024\]** We release **YuLan-Chat-12B-v3**, a chat-based LLM trained from scratch. It has been pre-trained on over 1.6TB English and Chinese corpus, and then supervised fine-tuned via curriculum learning with high-quality English and Chinese instructions and human preference data. 
* **\[Aug. 18, 2023\]** Our **YuLan-Chat-2-13B** achieves the 5th position of [OpenCompass](https://opencompass.org.cn/leaderboard-llm) benchmark!
* **\[Aug. 02, 2023\]** We release **YuLan-LLaMA-2-13B** and **YuLan-Chat-2-13B**. Both models have been continually pre-trained on English and Chinese corpus based on LLaMA-2, and YuLan-Chat-2-13B is the chat-based LLM based on YuLan-LLaMA-2-13B, with high-quality English and Chinese instructions.
* **\[Aug. 02, 2023\]** We release **YuLan-Chat-1-65B-v2**, a chat-based LLM based on LLaMA. It has been continually pre-trained on English and Chinese corpus, and then instruction-tuned with high-quality English and Chinese instructions.
* **\[Jun. 08, 2023\]** We release **YuLan-Chat-1-13B-v1** and **YuLan-Chat-1-65B-v1**, and the corresponding INT-8 quantization scripts. 

> * **\[2024年4月12日\]** 我们发布了**YuLan-Chat-12B-v3**模型，其通过完全从头开始训练得到，其通过在超过1.6TB的中英文数据上进行了大规模预训练, 然后基于高质量双语指令和人类偏好数据，使用课程学习方法进行有监督微调。
> * **\[2023年8月2日\]** 我们发布了**YuLan-LLaMA-2-13B**和**YuLan-Chat-2-13B**两个模型，其都在LLaMA-2的基础上进行了双语继续预训练，YuLan-Chat-2-13B在YuLan-LLaMA-2-13B基础上进行了双语高质量对话指令微调。
> * **\[2023年8月2日\]** 我们发布了**YuLan-Chat-1-65B-v2**模型，其在LLaMA-65B的基础上进行了双语继续预训练, 然后用高质量双语指令进行了微调。
> * **\[2023年6月8日\]** 我们发布了**YuLan-Chat-1-13B-v1**和**YuLan-Chat-1-65B-v1**两个模型，以及对应的int8量化脚本。

## Model Zoo

Due to the license limitation, for models based on LLaMA, we only provide the weight difference with the original checkpoints; for models based on LLaMA-2, they can be used directly. Please check the [Usage](https://github.com/RUC-GSAI/YuLan-Chat/tree/main#usage) section for more details.

**Limitations**: Despite our efforts to reduce potential security issues during the model's usage and encourage the generation of text that aligns with ethical and legal requirements, the language model is based on probabilistic generation, which means it may still produce unexpected outputs. For instance, the generated responses may contain biases, discrimination, or other harmful content. Please do not propagate such content. We do not assume any responsibility for any consequences resulting from the dissemination of harmful information.

> 由于许可证的限制，基于LLaMA的模型我们仅提供与官方模型的差值，基于LLaMA-2的模型可直接使用，具体请参见使用方法章节。

> **局限性**：尽管我们尝试减少模型在使用中可能出现的安全性问题，并鼓励模型生成符合道德和法律要求的文本，但由于语言模型基于概率生成的范式，模型仍然可能会产生意外的输出。 例如，生成的响应可能包含偏见、歧视或其他有害内容。 请不要传播此类内容。 我们对因传播有害信息而造成的任何后果不承担任何责任。

| Model               |  Backbone  | Extended Vocab | Extended Length | Continue PT | SFT  | Released Date |
| ------------------- | :--------: | :------------: | :-------------: | :---------: | ---- | :-----------: |
| YuLan-Chat-12B-v3 | YuLan-LLM-12B |    ✅ 51,190    |     ✅ 8,192     |      ✅      | ✅    |   2024.4.12    |
| [YuLan-Chat-2-13B](https://huggingface.co/yulan-team/YuLan-Chat-2-13b-fp16)    | LLaMA2-13B |    ✅ 51,190    |     ✅ 8,192     |      ✅      | ✅    |   2023.8.2    |
| [YuLan-LLaMA-2-13B](https://huggingface.co/yulan-team/YuLan-LLaMA-2-13b)     | LLaMA2-13B |    ✅ 51,190    |     ✅ 8,192     |      ✅      | ❌    |   2023.8.2    |
| [YuLan-Chat-1-65B-v2](https://huggingface.co/yulan-team/YuLan-Chat-1-65B-v2-delta) | LLaMA-65B  |    ✅ 51,190    |     ❌ 2,048     |      ✅      | ✅    |   2023.8.2    |
| [YuLan-Chat-1-13B-v1](https://huggingface.co/RUCAIBox/YuLan-Chat-13b-delta) | LLaMA-13B  |    ❌ 32,000    |     ❌ 2,048     |      ❌      |  ✅   |   2023.6.8    |
| [YuLan-Chat-1-65B-v1](https://huggingface.co/RUCAIBox/YuLan-Chat-65b-delta) | LLaMA-65B  |    ❌ 32,000    |     ❌ 2,048     |      ❌      |  ✅   |   2023.6.8    |

## Evaluation

We evaluate our YuLan-Chat model on several Chinese and English benchmarks. The evaluation results are shown as follows.

> 我们在中英文的一些基准测试上对YuLan-Chat进行了评价，其结果如下。

### MMLU

[MMLU](https://github.com/hendrycks/test) (Massive Multitask Language Understanding) is a benchmark designed to measure knowledge acquired during pretraining by evaluating models exclusively in zero-shot and few-shot settings.

> MMLU是一个评估模型知识量的常用的英文基准测试集。

| Model                             | STEM | Social Science | Humanities | Others | Avg. |
| --------------------------------- | :--: | :------------: | :--------: | :----: | :--: |
| YuLan-Chat-1-13B-v1               | 39.6 |      57.8      |    42.6    |  57.6  | 49.4 |
| YuLan-Chat-1-65B-v1               | 49.2 |      71.7      |    57.7    |  66.7  | 61.3 |
| YuLan-Chat-1-65B-v2               | 46.3 |      67.9      |    56.9    |  63.9  | 58.7 |
| LLaMA-2-13B                       | 44.6 |      64.2      |    53.9    |  62.2  | 56.2 |
| FlagAlpha/Llama2-Chinese-13b-Chat | 44.4 |      63.2      |    51.6    |  60.6  | 55.0 |
| Linly-AI/Chinese-LLaMA-2-13B-hf   | 43.6 |      62.7      |    49.8    |  61.6  | 54.4 |
| YuLan-LLaMA-2-13B                 | 42.9 |      61.5      |    50.4    |  58.6  | 53.4 |
| YuLan-Chat-2-13B                  | 45.3 |      66.7      |    53.8    |  62.8  | 57.2 |
### C-Eval

[C-Eval](https://cevalbenchmark.com/) is a comprehensive Chinese evaluation suite for foundation models.

> C-Eval是一个针对基石模型综合能力的中文基准测试集。

| Model                             | STEM | Social Science | Humanities | Others | Avg. | Avg. (Hard) |
| --------------------------------- | :--: | :------------: | :--------: | :----: | :--: | :---------: |
| YuLan-Chat-1-13B-v1               | 30.2 |      37.4      |    31.9    |  30.7  | 32.0 |    25.7     |
| YuLan-Chat-1-65B-v1               | 37.7 |      46.1      |    36.8    |  38.0  | 39.2 |    31.1     |
| YuLan-Chat-1-65B-v2               | 39.9 |      55.9      |    47.7    |  43.7  | 45.4 |    31.4     |
| LLaMA-2-13B                       | 36.9 |      43.2      |    37.6    |  36.6  | 38.2 |    32.0     |
| FlagAlpha/Llama2-Chinese-13b-Chat | 36.8 |      44.5      |    36.3    |  36.5  | 38.1 |    30.9     |
| Linly-AI/Chinese-LLaMA-2-13B-hf   | 33.7 |      44.8      |    36.6    |  36.5  | 37.0 |    27.7     |
| YuLan-LLaMA-2-13B                 | 35.3 |      46.4      |    41.9    |  37.6  | 39.3 |    28.6     |
| YuLan-Chat-2-13B                  | 38.9 |      49.7      |    45.0    |  40.8  | 42.6 |    32.2     |

### AGI-Eval-Gaokao

[AGI-Eval](https://github.com/microsoft/AGIEval) is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving. We use the sub-branch Chinese-Gaokao for evaluation.

> AGI-Eval 是一个以人为中心的基准，专门设计用于评估基础模型在与人类认知和解决问题相关的任务中的一般能力。我们使用其中的"高考"分支进行评测。

| Model                             | Avg. | Chinese | English | Geography | History | Biology | Chemistry | Physics | Math-QA | Math-Cloze |
| --------------------------------- | :--: | :-----: | :-----: | :-------: | :-----: | :-----: | :-------: | :-----: | :-----: | :--------: |
| YuLan-Chat-1-13B-v1               | 29.2 |  32.1   |  63.1   |   34.7    |  25.1   |  26.2   |   29.0    |  25.5   |  26.5   |    0.9     |
| YuLan-Chat-1-65B-v1               | 34.6 |  24.8   |  82.0   |   44.2    |  44.3   |  31.4   |   30.9    |  26.0   |  27.1   |    0.9     |
| YuLan-Chat-1-65B-v2               | 37.9 |  31.4   |  80.4   |   50.8    |  56.6   |  33.3   |   29.0    |  32.0   |  24.4   |    0.8     |
| LLaMA-2-13B                       | 32.7 |  27.2   |  72.2   |   36.2    |  43.0   |  26.2   |   32.4    |  30.0   |  26.2   |    0.9     |
| FlagAlpha/Llama2-Chinese-13b-Chat | 31.6 |  26.4   |  70.6   |   35.2    |  38.7   |  28.1   |   28.0    |  29.5   |  25.6   |    2.5     |
| Linly-AI/Chinese-LLaMA-2-13B-hf   | 31.1 |  22.8   |  74.8   |   42.2    |  37.9   |  24.3   |   28.0    |  23.0   |  26.5   |    0.0     |
| YuLan-LLaMA-2-13B                 | 34.2 |  25.2   |  70.3   |   43.2    |  48.5   |  30.0   |   29.5    |  31.0   |  28.5   |    1.7     |
| YuLan-Chat-2-13B                  | 39.5 |  37.0   |  85.3   |   46.7    |  51.9   |  43.8   |   38.2    |  29.0   |  23.1   |    0.9     |

## Usage

### Environment Setting

```
conda create -n yulan python=3.10 -y
conda activate yulan
```
We suggest to install the pytorch and bitsandbytes according to their official guidance for better adapting to your environment, and we provide our applied versions as reference:
> 我们建议根据官方手册安装pytorch和bitsandbytes，此处提供我们使用的版本作为参考。
```
torch==1.13
bitsandbytes==0.39.0
```
Then, you can install other packages by the following instruction: 
> 然后，安装其他所需的包。
```
pip install -r requirements.txt
```

### Model Weights Recovering

1. For YuLan-Chat-1-13B-v1, YuLan-Chat-1-65B-v1, and YuLan-Chat-1-65B-v2, as they are based on LLaMA, you should download [LLaMA](https://github.com/facebookresearch/llama)'s original weights, and then add our released delta parameters into the original parameters to compose the final model parameters.
> 对于基于LLaMA的模型，请先下载LLaMA官方模型，然后将我们发布的参数差值合并到原始模型参数中以获得最终的参数。
```
python3 apply_delta.py \
    --base-model-path ./llama-13b/ \
    --tuned-model-path ./yulan-13b/ \
    --delta-path ./yulan-13b-delta
```

2. For YuLan-LLaMA-2-13B and YuLan-Chat-2-13B, you can just download our released checkpoints and load their parameters via Huggingface Transformers.
> 对于基于LLaMA-2的模型，可以直接下载我们发布的模型权重，并使用Huggingface Transformers进行使用。

### Import from Huggingface Transformers

As our model is trained based on LLaMA, it can be loaded in the same way as original LLaMA.

> 由于我们的模型是基于LLaMA开发的，可以使用与LLaMA相同的方法加载。

```Python
>>> from transformers import LlamaTokenizer, LlamaForCausalLM
>>> tokenizer = LlamaTokenizer.from_pretrained("yulan-team/YuLan-Chat-2-13b")
>>> model = LlamaForCausalLM.from_pretrained("yulan-team/YuLan-Chat-2-13b").cuda()
>>> model = model.eval()
>>> input_text = "hello"
>>> prompt = "The following is a conversation between a human and an AI assistant namely YuLan, developed by GSAI, Renmin University of China. The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n[|Human|]:{}\n[|AI|]:".format(input_text)
>>> inputs = tokenizer(prompt, return_tensors='pt', padding="longest", max_length=8192, truncation=True, return_attention_mask=True, add_special_tokens=True)
>>> kwargs = {'temperature': 0.8, 'top_p': 0.95, "top_k": 50, "repetition_penalty": 1.1, "no_repeat_ngram_size": 64, "max_length": 8192, "pad_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id}
>>> outputs = model.generate(inputs['input_ids'].to(model.device), attention_mask=inputs['attention_mask'].to(model.device), do_sample=True, **kwargs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[len(prompt):])
Hello! How can I assist you today?
```

### Inference in Command Line

We provide the code for the inference of YuLan-Chat in command line.
> 我们提供命令行预测脚本。
```
python inference.py --model_path ~/pretrain-checkpoint/yulan-13b/
```

We also provide a quantization way for efficiently deploying YuLan-Chat. After quantization, YuLan-Chat can be loaded into a single GPU.
> 我们也提供了一种量化的方法以便于更轻量化地部署YuLan-Chat。经过量化后，模型可以被加载进单张GPU中。

|YuLan-Chat (INT-8)| GPU Consumption |
|------------------|-----------------|
|13B| RTX3090-24G    |
|65B| A100-80G       |
```
python inference.py --model_path ~/pretrain-checkpoint/yulan-13b/ --load_in_8bit
```


## License

YuLan-Chat uses [MIT License](https://github.com/RUC-GSAI/YuLan-Chat/blob/main/LICENSE). All data and code in this project can only be used for academic purposes.

> 本项目使用MIT许可，所有的数据和代码仅供学术研究使用。

## Contributors

|       **Pre-training**              | **Fine-tuning**                                                 |
|:----------------------------- |:-------------------------------------------------------------------- |
| [Yutao Zhu](https://github.com/DaoD) (Lead), [Kelong Mao](https://github.com/kyriemao), [Wentong Chen](https://github.com/yiye3), [Yiding Sun](https://github.com/Emanual20), [Yihan Wu](https://github.com/wyh2000), [Qian Cao](https://github.com/Aman-4-Real), [Lei Zhang](https://github.com/LLily0703), [Feng Wang](https://github.com/PhealenWang), [Qiangqiang Ren](https://github.com/QiangKing)| [Kun Zhou](https://github.com/Lancelot39) (Lead), [Yushuo Chen](https://github.com/chenyushuo), [Zhipeng Chen](https://github.com/Timothy023), [Lei Wang](https://github.com/Paitesanshi), [Yupeng Hou](https://github.com/hyp1231), [Xincheng Pang](https://github.com/pangxincheng), [Xinyu Tang](https://github.com/txy77), [Junyi Li](https://github.com/turboLJY), [Yuhan Chen](https://github.com/Fiorina1212), [Shufang Xie](https://github.com/funtion) |

## Reference

Please kindly cite our work if it helps you.

> 如果我们的项目对您有帮助，请引用我们，谢谢！

```BibTeX
@misc{YuLan-Chat,
  author = {YuLan-Team},
  title = {YuLan-Chat: An Open-Source Bilingual Chatbot},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/RUC-GSAI/YuLan-Chat}},
}
```

## YuLan-1

You can refer to our [original branch](https://github.com/RUC-GSAI/YuLan-Chat/tree/YuLan-Chat-1) for more detail about YuLan-Chat-1 and the instruction collection.
> 更多关于指令构造的细节，可以参考我们之前的分支。

## Star History

<!--     <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=RUC-GSAI/YuLan-Chat&type=Date&theme=dark" /> -->
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=RUC-GSAI/YuLan-Chat&type=Date" />
