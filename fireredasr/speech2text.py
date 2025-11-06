#!/usr/-bin/env python3

import argparse
import glob
import os
import sys

from fireredasr.models.fireredasr import FireRedAsr


parser = argparse.ArgumentParser()
parser.add_argument('--asr_type', type=str, required=True, choices=["aed", "llm"])
parser.add_argument('--model_dir', type=str, required=True)

# Input / Output
parser.add_argument("--wav_path", type=str)
parser.add_argument("--wav_paths", type=str, nargs="*")
parser.add_argument("--wav_dir", type=str)
parser.add_argument("--wav_scp", type=str)
parser.add_argument("--output", type=str)

# Model Configuration
parser.add_argument('--asr_device', type=str, default="cuda:0", help="Device for AED model or the ASR part of the LLM model.")
parser.add_argument('--llm_device', type=str, default="cuda:1", help="Device for the LLM part. Can be 'cpu', 'cuda:1', etc.")
parser.add_argument('--llm_dtype', type=str, default="fp16", choices=["fp32", "fp16", "bf16"], help="Data type for LLM inference (float32, float16, bfloat16).")
parser.add_argument('--use_flash_attn', type=int, default=1, choices=[0, 1], help="Whether to use flash attention 2 for the LLM (1 for True, 0 for False).")
parser.add_argument('--use_gpu', type=int, default=1, help="Global GPU switch for AED mode if asr_device is not specified.")

# Inference Parameters
parser.add_argument("--batch_size", type=int, default=1, help="Number of audio files to process in parallel during model inference.")
parser.add_argument("--beam_size", type=int, default=1)
parser.add_argument("--decode_max_len", type=int, default=0)
# FireRedASR-AED
parser.add_argument("--nbest", type=int, default=1)
parser.add_argument("--softmax_smoothing", type=float, default=1.0)
parser.add_argument("--aed_length_penalty", type=float, default=0.0)
parser.add_argument("--eos_penalty", type=float, default=1.0)
# FireRedASR-LLM
parser.add_argument("--decode_min_len", type=int, default=0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--llm_length_penalty", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=1.0)


def main(args):
    wavs = get_wav_info(args)
    fout = open(args.output, "w") if args.output else None

    # 1. 初始化时传入模型配置参数
    model = FireRedAsr.from_pretrained(
        asr_type=args.asr_type,
        model_dir=args.model_dir,
        asr_device=args.asr_device,
        llm_device=args.llm_device,
        llm_dtype=args.llm_dtype,
        use_flash_attn=bool(args.use_flash_attn),
        use_gpu=bool(args.use_gpu)
    )

    # 提取所有音频文件的路径
    all_wav_paths = [wav_info[1] for wav_info in wavs]

    # 2. 一次性调用transcribe处理所有文件
    # 内部会根据batch_size自动分批并显示进度条
    results = model.transcribe(
        all_wav_paths,
        batch_size=args.batch_size,
        **vars(args)
    )

    # 3. 处理所有返回的结果
    for result in results:
        print(result)
        if fout is not None:
            fout.write(f"{result['uttid']}\t{result['text']}\n")


def get_wav_info(args):
    """
    Returns:
        wavs: list of (uttid, wav_path)
    """
    base = lambda p: os.path.splitext(os.path.basename(p))[0]
    if args.wav_path:
        wavs = [(base(args.wav_path), args.wav_path)]
    elif args.wav_paths and len(args.wav_paths) >= 1:
        wavs = [(base(p), p) for p in sorted(args.wav_paths)]
    elif args.wav_scp:
        wavs = [line.strip().split(maxsplit=1) for line in open(args.wav_scp)]
    elif args.wav_dir:
        wavs = glob.glob(f"{args.wav_dir}/**/*.wav", recursive=True)
        wavs = [(base(p), p) for p in sorted(wavs)]
    else:
        raise ValueError("Please provide valid wav info")
    print(f"#wavs={len(wavs)}")
    return wavs


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)