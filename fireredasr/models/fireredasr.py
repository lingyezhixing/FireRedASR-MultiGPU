import os
import time

import torch

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper


class FireRedAsr:
    @classmethod
    def from_pretrained(
        cls,
        asr_type,
        model_dir,
        asr_device="cuda:0",
        llm_device="cpu",
        llm_dtype="fp16",
        use_flash_attn=False, # <-- 新增参数
        use_gpu=True,
    ):
        assert asr_type in ["aed", "llm"]

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path = os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)

            device = asr_device if use_gpu else "cpu"
            model.to(device)

        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir,
                asr_device=asr_device,
                llm_device=llm_device,
                llm_dtype=llm_dtype,
                use_flash_attn=use_flash_attn # <-- 传递参数
            )

        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer)

    def __init__(self, asr_type, feat_extractor, model, tokenizer):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path, **kwargs):
        feats, lengths, durs = self.feat_extractor(batch_wav_path)
        total_dur = sum(durs)

        if self.asr_type == "aed":
            device = next(self.model.parameters()).device
            feats, lengths = feats.to(device), lengths.to(device)

        start_time = time.time()
        if self.asr_type == "aed":
            hyps = self.model.transcribe(
                feats, lengths,
                beam_size=kwargs.get("beam_size", 1),
                nbest=kwargs.get("nbest", 1),
                decode_max_len=kwargs.get("decode_max_len", 0),
                softmax_smoothing=kwargs.get("softmax_smoothing", 1.0),
                aed_length_penalty=kwargs.get("aed_length_penalty", 0.0),
                eos_penalty=kwargs.get("eos_penalty", 1.0)
            )
            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0

            results = []
            for uttid, wav, hyp in zip(batch_uttid, batch_wav_path, hyps):
                hyp = hyp[0]
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                results.append({"uttid": uttid, "text": text, "wav": wav, "rtf": f"{rtf:.4f}"})
            return results

        elif self.asr_type == "llm":
            input_ids, attention_mask, _, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""] * feats.size(0), tokenizer=self.tokenizer,
                    max_len=128, decode=True)

            generated_ids = self.model.transcribe(
                padded_feat=feats,
                feat_lengths=lengths,
                padded_input_ids=input_ids,
                attention_mask=attention_mask,
                beam_size=kwargs.get("beam_size", 1),
                decode_max_len=kwargs.get("decode_max_len", 0),
                decode_min_len=kwargs.get("decode_min_len", 0),
                repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                llm_length_penalty=kwargs.get("llm_length_penalty", 0.0),
                temperature=kwargs.get("temperature", 1.0)
            )
            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0
            texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                results.append({"uttid": uttid, "text": text, "wav": wav, "rtf": f"{rtf:.4f}"})
            return results


def load_fireredasr_aed_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model


def load_firered_llm_model_and_tokenizer(
    model_path, encoder_path, llm_dir,
    asr_device, llm_device, llm_dtype, use_flash_attn # <-- 接收新参数
):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    args = package["args"]
    args.encoder_path = encoder_path
    args.llm_dir = llm_dir

    # 将所有配置参数添加到模型参数命名空间中
    args.asr_device = asr_device
    args.llm_device = llm_device
    args.llm_dtype = llm_dtype
    args.use_flash_attn = use_flash_attn # <-- 添加新参数

    print("model args:", args)
    model = FireRedAsrLlm.from_args(args)
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer