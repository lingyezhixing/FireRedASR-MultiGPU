from fireredasr.models.fireredasr import FireRedAsr

all_wav_paths = [
    r"audio_split_output_vad\gaowanying_chunk_0001.wav",
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
    use_flash_attn=False  # <-- 新增配置项
)

# 2. 推理时传入所有音频路径和解码参数
# 新增的 `batch_size` 参数用于控制 `transcribe` 方法内部的批处理大小。
# 这对于在处理大量文件时防止小显存设备内存溢出非常有用。
# 对于16GB显存的设备，建议将 `batch_size` 设置为1，`beam_size`设置为3。此时可以处理最长16s的音频文件。
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