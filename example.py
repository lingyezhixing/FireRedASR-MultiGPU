from fireredasr.models.fireredasr import FireRedAsr

batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav"]

# # FireRedASR-AED
# # 1. 初始化时配置模型
# model = FireRedAsr.from_pretrained(
#     asr_type="aed",
#     model_dir="pretrained_models/FireRedASR-AED-L",
#     asr_device="cuda:0"
# )
#
# # 2. 推理时传入解码参数
# results = model.transcribe(
#     batch_uttid,
#     batch_wav_path,
#     beam_size=3,
#     nbest=1,
#     decode_max_len=0,
#     softmax_smoothing=1.25,
#     aed_length_penalty=0.6,
#     eos_penalty=1.0
# )
# print(results)


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

# 2. 推理时只传入与解码相关的参数
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    beam_size=3,
    decode_max_len=0,
    decode_min_len=0,
    repetition_penalty=3.0,
    llm_length_penalty=1.0,
    temperature=1.0
)
print(results)