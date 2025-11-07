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
如果你想使用 `FireRedASR-LLM-L`，还需要下载 [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) 并将其放在 `pretrained_models` 文件夹中。

### 安装设置
创建 Python 环境并安装依赖项
```bash
$ git clone https://github.com/lingyezhixing/FireRedASR.git
$ conda create --name fireredasr python=3.12
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

### Python 使用方式
```python
from fireredasr.models.fireredasr import FireRedAsr

all_wav_paths = [
    r"audio.wav",
]

# FireRedASR-AED
# 1. 初始化时配置模型
model = FireRedAsr.from_pretrained(
    asr_type="aed",
    model_dir="pretrained_models/FireRedASR-AED-L",
    asr_device="cuda:0"
)
# 2. 推理时传入解码参数，并指定内部批处理大小
results = model.transcribe(
    all_wav_paths,
    batch_size=4,
    beam_size=3,
    nbest=1,
    decode_max_len=0,
    softmax_smoothing=1.25,
    aed_length_penalty=0.6,
    eos_penalty=1.0
)
print(results)


# FireRedASR-LLM
# 1. 初始化时进行设备、数据类型和 Flash Attention 配置
model = FireRedAsr.from_pretrained(
    asr_type="llm",
    model_dir="pretrained_models/FireRedASR-LLM-L",
    asr_device="cuda:0",
    llm_device="cuda:1",
    llm_dtype="bf16",
    use_flash_attn=False
)

# 2. 推理时传入所有音频路径和解码参数
# 新增的 `batch_size` 参数用于控制 `transcribe` 方法内部的批处理大小。
# 这对于在处理大量文件时防止小显存设备内存溢出非常有用。
# 对于16GB显存的设备，建议将 `batch_size` 设置为1。此时在不启用 `use_flash_attn` 的情况下最长可以处理16s的音频文件。
results = model.transcribe(
    all_wav_paths,
    batch_size=1,
    beam_size=3,
    decode_max_len=0,
    decode_min_len=0,
    repetition_penalty=3.0,
    llm_length_penalty=1.0,
    temperature=1.0
)
print(results)
```

# API 文档（FireRedASR.py）
**版本: 1.3.1**

## 概述
本文档详细说明了如何使用 FireRedASR API 来进行高精度的语音识别。该API服务提供了一个统一的端点，支持处理单个或批量的音频文件，并允许对底层ASR模型及解码参数进行全面、灵活的配置。
服务器端对模型路径进行了统一管理，调用者只需通过 `asr_type` 参数选择 `aed` 或 `llm` 模型，无需关心其物理存储位置。

## 端点: `POST /transcribe/FireRedASR`
这是执行语音转文字的核心API端点。

### **并发处理机制 (重要)**
为了保证服务的稳定性和避免因资源竞争（尤其是GPU显存）导致的崩溃，本API服务器内部采用了 **队列处理机制**。
-   服务器可以 **同时接收** 多个并发请求。
-   所有接收到的请求任务会进入一个 **先进先出（FIFO）的队列**。
-   服务器会 **按顺序、逐一** 从队列中取出任务进行处理。
-   只有当前任务完全处理结束后，下一个任务才会开始。
这意味着，对于API使用者来说，即使同时发送多个请求，它们的处理也是顺序的。在高负载情况下，后续请求的等待时间可能会增加，但这种设计确保了每个任务都能在资源充足的环境下被正确、稳定地执行。

### **健康检查端点: `GET /health`**
用于检测服务是否正常运行。
-   **响应**: `{"status": "healthy"}`
-   **状态码**: 200 OK

### 请求格式
请求必须采用 `multipart/form-data` 格式。您可以通过以下两种方式之一提供音频：
1.  **`files`**: 直接上传一个或多个音频文件。单个请求中上传的所有文件会被视为一个处理批次。
2.  **`paths`**: 提供一个以逗号分隔的字符串，内容是位于服务器上的音频文件的绝对路径。

**注意:** 在单次请求中，您必须提供 `files` 或 `paths` 之一，但不能同时提供。

### 表单参数 (Form Parameters)
所有参数都以表单字段（Form Fields）的形式发送。

#### **模型加载参数**
这些参数决定了后端加载哪个模型以及如何配置它。当这些参数发生变化时，服务器会自动重新加载模型。可以全部不指定，默认使用 `FireRedASR-LLM-L`和默认配置。

| 参数             | 类型    | 默认值   | 可选值                          | 描述                                                                    |
| :--------------- | :------ | :------- | :------------------------------ | :---------------------------------------------------------------------- |
| `asr_type`       | 字符串  | `llm`    | `aed`, `llm`                    | 模型架构。                                                              |
| `asr_device`     | 字符串  | `cuda:0` | -                               | ASR编码器所用的设备 (例如 `cuda:0`, `cpu`)。                            |
| `llm_device`     | 字符串  | `cuda:1` | -                               | LLM部分所用的设备 (例如 `cuda:1`, `cpu`)。仅在 `asr_type=llm` 时使用。 |
| `llm_dtype`      | 字符串  | `bf16`   | `fp32`, `fp16`, `bf16`          | LLM推理时的数据类型。                                                   |
| `use_flash_attn` | 布尔值  | `false`  | `true`, `false`                 | 设置为 `true` 以启用LLM的 Flash Attention 2。                           |

#### **转录参数**
这些参数控制解码过程的行为，可以在不重新加载模型的情况下动态调整，可以不指定，会自动使用默认配置。

| 参数                 | 类型    | 默认值 | 描述                                                                    |
| :------------------- | :------ | :----- | :---------------------------------------------------------------------- |
| `batch_size`         | 整数    | `1`    | 推理时的内部批处理大小，用于控制显存（VRAM）使用量。                      |
| `beam_size`          | 整数    | `3`    | 解码过程中使用的束搜索（Beam Search）大小。                               |
| `decode_max_len`     | 整数    | `0`    | 生成文本序列的最大长度。`0` 表示不设限制。                              |
| `nbest`              | 整数    | `1`    | **(AED专用)** 返回的最佳假设的数量。                                    |
| `softmax_smoothing`  | 浮点数  | `1.0`  | **(AED专用)** Softmax平滑因子。                                         |
| `aed_length_penalty` | 浮点数  | `0.6`  | **(AED专用)** 对较长序列的惩罚系数。                                    |
| `eos_penalty`        | 浮点数  | `1.0`  | **(AED专用)** 对句末符（EOS）的惩罚系数。                               |
| `decode_min_len`     | 整数    | `0`    | **(LLM专用)** 生成文本序列的最小长度。                                  |
| `repetition_penalty` | 浮点数  | `3.0`  | **(LLM专用)** 用于抑制重复词语的惩罚系数。                              |
| `llm_length_penalty` | 浮点数  | `1.0`  | **(LLM专用)** 对较长序列的惩罚系数。                                    |
| `temperature`        | 浮点数  | `1.0`  | **(LLM专用)** 采样温度。`1.0` 表示不改变。                              |

### 响应格式

#### **成功响应 (200 OK)**
API会返回一个JSON数组，其中每个对象对应一个已转录的音频文件。
-   **`uttid`**: 唯一标识符，从文件名派生而来。
-   **`text`**: 模型识别出的文本结果。
-   **`wav`**: 被处理的音频文件路径。如果是文件上传，这里会显示其在服务器上的临时路径（绝对路径）。如果是路径上传，这里会显示你提供的路径（不会转换为绝对路径，你提供什么就返回什么）。
-   **`rtf`**: 实时率（Real-Time Factor），表示处理时长与音频时长的比值，越小越好。

**响应示例:**
```json
[
  {
    "uttid": "audio_chunk_01",
    "text": "这里是第一段音频的转录文本内容",
    "wav": "Temp/tmp_xyz_audio_chunk_01.wav",
    "rtf": "0.1588"
  },
  {
    "uttid": "audio_chunk_02",
    "text": "这是第二段音频的识别结果",
    "wav": "Temp/tmp_abc_audio_chunk_02.wav",
    "rtf": "0.1601"
  }
]
```

#### **错误响应**
-   **`400 Bad Request`**: 如果请求中既未提供 `files` 也未提供 `paths`，或服务器未配置请求的 `asr_type`，则返回此错误。
-   **`422 Unprocessable Entity`**: 如果任何表单参数未能通过验证（例如，`asr_type` 不是 `aed` 或 `llm`，或 `batch_size` 不是正整数），则返回此错误。响应体中将包含详细的验证错误信息。
-   **`500 Internal Server Error`**: 如果服务器端配置的模型路径不存在，则返回此错误。

## 完整请求示例

以下示例展示了如何通过 `curl` 和 Python `requests` 库调用此API。

### 示例1：上传单个文件，使用默认LLM模型
#### **使用 `curl`**
```bash
# -F "files=@/path/to/your/audio.wav" 表示上传一个文件
# 其他参数使用默认值
curl -X POST "http://127.0.0.1:8000/transcribe/FireRedASR" \
  -F "files=@/data/my_audio/meeting_part_1.wav" \
  -H "accept: application/json"
```

#### **使用 Python `requests`**
```python
import requests

# API的URL
url = "http://127.0.0.1:8000/transcribe/FireRedASR"

# 要上传的音频文件路径
file_path = "/data/my_audio/meeting_part_1.wav"

# 'rb' 表示以二进制读取模式打开文件
with open(file_path, 'rb') as f:
    # 准备文件部分，'files' 键需要与API中定义的参数名匹配
    files = {'files': (file_path, f, 'audio/wav')}
    
    # 发送POST请求
    response = requests.post(url, files=files)

# 检查响应
if response.status_code == 200:
    print("请求成功!")
    print("识别结果:", response.json())
else:
    print(f"请求失败，状态码: {response.status_code}")
    print("错误详情:", response.text)
```

### 示例2：上传多个文件，使用AED模型并自定义参数
#### **使用 `curl`**
```bash
# 多次使用 -F "files=@..." 来上传多个文件
# 同时通过 -F 传递其他自定义参数
# 注意：不再需要传递 model_dir
curl -X POST "http://127.0.0.1:8000/transcribe/FireRedASR" \
  -H "accept: application/json" \
  -F "files=@/audios/chunk_001.wav" \
  -F "files=@/audios/chunk_002.wav" \
  -F "asr_type=aed" \
  -F "batch_size=2" \
  -F "beam_size=5" \
  -F "aed_length_penalty=0.8"
```

#### **使用 Python `requests`**
```python
import requests

url = "http://127.0.0.1:8000/transcribe/FireRedASR"

# 要上传的多个文件路径
file_paths = ["/audios/chunk_001.wav", "/audios/chunk_002.wav"]

# 准备文件列表
# 对于多个文件，files参数需要是一个元组列表
files_list = [
    ('files', (path, open(path, 'rb'), 'audio/wav')) for path in file_paths
]

# 准备其他表单参数
# 注意：不再需要传递 model_dir
payload = {
    'asr_type': 'aed',
    'batch_size': 2,
    'beam_size': 5,
    'aed_length_penalty': 0.8
}

# 发送请求，同时传递文件和数据
response = requests.post(url, files=files_list, data=payload)

if response.status_code == 200:
    print("请求成功!")
    print("识别结果:", response.json())
else:
    print(f"请求失败，状态码: {response.status_code}")
    print("错误详情:", response.text)
```

### 示例3：指定服务器上的多个文件路径
#### **使用 `curl`**
```bash
# -F "paths=..." 传递一个逗号分隔的路径字符串
curl -X POST "http://127.0.0.1:8000/transcribe/FireRedASR" \
  -H "accept: application/json" \
  -F "paths=/server/audio/file_a.wav,/server/audio/file_b.wav" \
  -F "batch_size=2"
```

#### **使用 Python `requests`**
```python
import requests

url = "http://127.0.0.1:8000/transcribe/FireRedASR"

# 服务器上的文件路径，用逗号连接
server_paths = "/server/audio/file_a.wav,/server/audio/file_b.wav"

# 准备数据负载
payload = {
    'paths': server_paths,
    'batch_size': 2
}

# 此时不需要files参数，只需要data参数
response = requests.post(url, data=payload)

if response.status_code == 200:
    print("请求成功!")
    print("识别结果:", response.json())
else:
    print(f"请求失败，状态码: {response.status_code}")
    print("错误详情:", response.text)
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