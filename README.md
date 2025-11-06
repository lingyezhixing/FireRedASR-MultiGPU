<div align="center">
<h1>FireRedASR：开源工业级自动语音识别模型</h1>
</div>
[[论文]](https://arxiv.org/pdf/2501.14350)
[[模型]](https://huggingface.co/fireredteam)
[[博客]](https://fireredteam.github.io/demos/firered_asr/)
[[演示]](https://huggingface.co/spaces/FireRedTeam/FireRedASR)

FireRedASR 是一系列开源的工业级自动语音识别（ASR）模型，支持中文、中文方言和英文，达到了公开中文 ASR 基准上的新最先进水平（SOTA），同时在歌曲歌词识别方面也表现出色。

## 🔥 最新动态
- [2025/02/17] 我们发布了 [FireRedASR-LLM-L](https://huggingface.co/fireredteam/FireRedASR-LLM-L/tree/main) 模型权重。
- [2025/01/24] 我们发布了[技术报告](https://arxiv.org/pdf/2501.14350)、[博客](https://fireredteam.github.io/demos/firered_asr/)以及 [FireRedASR-AED-L](https://huggingface.co/fireredteam/FireRedASR-AED-L/tree/main) 模型权重。

## 方法
FireRedASR 设计用于满足各种应用场景中对高性能与高效率的需求。它包含两个变体：
- FireRedASR-LLM：旨在实现最先进的性能并支持无缝端到端语音交互。该模型采用编码器-适配器-大语言模型（LLM）框架，利用大语言模型的能力。
- FireRedASR-AED：在高性能与计算效率之间取得平衡，并作为基于 LLM 的语音模型中的有效语音表示模块。它使用基于注意力机制的编码器-解码器（AED）架构。
![Model](/assets/FireRedASR_model.png)

## 评估
结果以字符错误率（CER%）表示中文，以单词错误率（WER%）表示英文。
### 公开中文 ASR 基准上的评估
| 模型            | 参数量 | aishell1 | aishell2 | ws\_net  | ws\_meeting | 平均-4 |
|:----------------:|:-------:|:--------:|:--------:|:--------:|:-----------:|:---------:|
| FireRedASR-LLM   | 8.3B | 0.76 | 2.15 | 4.60 | 4.67 | 3.05 |
| FireRedASR-AED   | 1.1B | 0.55 | 2.52 | 4.88 | 4.76 | 3.18 |
| Seed-ASR         | 12B+ | 0.68 | 2.27 | 4.66 | 5.69 | 3.33 |
| Qwen-Audio       | 8.4B | 1.30 | 3.10 | 9.50 | 10.87 | 6.19 |
| SenseVoice-L     | 1.6B | 2.09 | 3.04 | 6.01 | 6.73 | 4.47 |
| Whisper-Large-v3 | 1.6B | 5.14 | 4.96 | 10.48 | 18.87 | 9.86 |
| Paraformer-Large | 0.2B | 1.68 | 2.85 | 6.74 | 6.97 | 4.56 |
`ws` 表示 WenetSpeech。

### 公开中文方言和英文 ASR 基准上的评估
| 测试集       | KeSpeech | LibriSpeech test-clean | LibriSpeech test-other  |
| :------------:| :------: | :--------------------: | :----------------------:|
|FireRedASR-LLM | 3.56 | 1.73 | 3.67 |
|FireRedASR-AED | 4.48 | 1.93 | 4.44 |
|以往最先进结果   | 6.70 | 1.82 | 3.50 |

## 使用方式
从 [HuggingFace](https://huggingface.co/fireredteam) 下载模型文件并将其放置在 `pretrained_models` 文件夹中。
如果你想使用 `FireRedASR-LLM-L`，还需要下载 [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) 并将其放在 `pretrained_models` 文件夹中。然后进入 `FireRedASR-LLM-L` 文件夹并运行命令：`$ ln -s ../Qwen2-7B-Instruct`

### 安装设置
创建 Python 环境并安装依赖项
```bash
$ git clone https://github.com/FireRedTeam/FireRedASR.git
$ conda create --name fireredasr python=3.10
$ pip install -r requirements.txt
```
设置 Linux 路径和 PYTHONPATH：
```
$ export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
$ export PYTHONPATH=$PWD/:$PYTHONPATH
```
将音频转换为 16kHz、16位 PCM 格式：
```
ffmpeg -i input_audio -ar 16000 -ac 1 -acodec pcm_s16le -f wav output.wav
```

### 快速开始
```bash
$ cd examples
$ bash inference_fireredasr_aed.sh
$ bash inference_fireredasr_llm.sh
```

### 命令行使用方式
```bash
$ speech2text.py --help
$ speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav --asr_type "aed" --model_dir pretrained_models/FireRedASR-AED-L
$ speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav --asr_type "llm" --model_dir pretrained_models/FireRedASR-LLM-L
```

### Python 使用方式
```python
from fireredasr.models.fireredasr import FireRedAsr
batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav"]
# FireRedASR-AED
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,
        "beam_size": 3,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.25,
        "aed_length_penalty": 0.6,
        "eos_penalty": 1.0
    }
)
print(results)

# FireRedASR-LLM
model = FireRedAsr.from_pretrained("llm", "pretrained_models/FireRedASR-LLM-L")
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,
        "beam_size": 3,
        "decode_max_len": 0,
        "decode_min_len": 0,
        "repetition_penalty": 3.0,
        "llm_length_penalty": 1.0,
        "temperature": 1.0
    }
)
print(results)
```

## 使用提示
### 批量束搜索（Batch Beam Search）
- 在使用 FireRedASR-LLM 进行批量束搜索时，请确保输入语音的长度相近。如果语音长度差异较大，较短的语音可能会出现重复问题。你可以通过按长度排序数据集或设置 `batch_size` 为 1 来避免此问题。
### 输入长度限制
- FireRedASR-AED 支持最长 60 秒的音频输入。超过 60 秒可能导致幻觉问题，超过 200 秒将引发位置编码错误。
- FireRedASR-LLM 支持最长 30 秒的音频输入。对于更长的输入行为目前未知。

## 致谢
感谢以下开源项目的贡献：
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [icefall/ASR_LLM](https://github.com/k2-fsa/icefall/tree/master/egs/speech_llm/ASR_LLM)
- [WeNet](https://github.com/wenet-e2e/wenet)
- [Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)

## 引用
```bibtex
@article{xu2025fireredasr,
  title={FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration},
  author={Xu, Kai-Tuo and Xie, Feng-Long and Tang, Xu and Hu, Yao},
  journal={arXiv preprint arXiv:2501.14350},
  year={2025}
}
```
```