import logging
import os
import random
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.module.adapter import Adapter
from fireredasr.tokenizer.llm_tokenizer import DEFAULT_SPEECH_TOKEN, IGNORE_TOKEN_ID
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper
from fireredasr.utils.param import count_model_parameters


class FireRedAsrLlm(nn.Module):
    @classmethod
    def load_encoder(cls, model_path):
        assert os.path.exists(model_path)
        package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        model = FireRedAsrAed.from_args(package["args"])
        if "model_state_dict" in package:
            model.load_state_dict(package["model_state_dict"], strict=False)
        encoder = model.encoder
        encoder_dim = encoder.odim
        return encoder, encoder_dim

    @classmethod
    def from_args(cls, args):
        logging.info(args)
        logging.info("Build FireRedAsrLlm")

        # 1. 获取配置信息
        asr_device = getattr(args, 'asr_device', 'cuda:0')
        llm_device = getattr(args, 'llm_device', 'cpu')
        llm_dtype_str = getattr(args, 'llm_dtype', 'fp16')
        use_flash_attn = getattr(args, 'use_flash_attn', False) # <-- 此处已能正确获取

        dtype_map = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32
        }
        torch_dtype = dtype_map.get(llm_dtype_str, torch.float16)

        if torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logging.warning("bfloat16 is not supported on this device. Falling back to float16.")
            torch_dtype = torch.float16

        logging.info(f"Initializing model on devices -> ASR Encoder/Projector: {asr_device}, LLM: {llm_device}")
        logging.info(f"LLM will be loaded with dtype: {torch_dtype}")
        logging.info(f"Using Flash Attention 2: {use_flash_attn}")

        # 2. 构建并放置 Speech Encoder
        encoder, encoder_dim = cls.load_encoder(args.encoder_path)
        encoder.to(asr_device)
        count_model_parameters(encoder)
        if args.freeze_encoder:
            logging.info(f"Frezee encoder")
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            encoder.eval()

        attn_implementation = "flash_attention_2" if use_flash_attn else "eager"

        # 3. 构建并放置 LLM
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        llm.to(llm_device)
        count_model_parameters(llm)

        # LLM Freeze or LoRA
        llm_dim = llm.config.hidden_size
        if args.freeze_llm:
            logging.info(f"Frezee LLM")
            for name, param in llm.named_parameters():
                param.requires_grad = False
            llm.eval()
        else:
            if getattr(args, 'use_lora', False):
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                )
                llm = get_peft_model(llm, lora_config)
                llm.print_trainable_parameters()

        tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(args.llm_dir)
        llm.config.pad_token_id = tokenizer.pad_token_id
        llm.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        llm.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)

        # 4. 构建并放置 projector
        encoder_projector = Adapter(encoder_dim, llm_dim, args.encoder_downsample_rate)
        encoder_projector.to(asr_device)
        count_model_parameters(encoder_projector)

        return cls(encoder, llm, encoder_projector,
                   args.freeze_encoder, args.freeze_llm)

    def __init__(self, encoder, llm, encoder_projector,
                 freeze_encoder, freeze_llm):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.freeze_encoder = freeze_encoder
        self.freeze_llm = freeze_llm
        self.llm_config = llm.config
        self.asr_device = next(encoder.parameters()).device
        self.llm_device = next(llm.parameters()).device

    def transcribe(
        self,
        padded_feat,
        feat_lengths,
        padded_input_ids,
        attention_mask,
        beam_size=1,
        decode_max_len=0,
        decode_min_len=0,
        repetition_penalty=1.0,
        llm_length_penalty=1.0,
        temperature=1.0
    ):
        padded_feat = padded_feat.to(self.asr_device)
        feat_lengths = feat_lengths.to(self.asr_device)
        encoder_outs, enc_lengths, enc_mask = self.encoder(padded_feat, feat_lengths)
        speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)

        speech_features = speech_features.to(device=self.llm_device, dtype=self.llm.dtype)

        padded_input_ids = padded_input_ids.to(self.llm_device)
        attention_mask = attention_mask.to(self.llm_device)
        inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)

        inputs_embeds, attention_mask, _ = \
            self._merge_input_ids_with_speech_features(
                speech_features, inputs_embeds, padded_input_ids, attention_mask,
                speech_lens=speech_lens
            )

        max_new_tokens = speech_features.size(1) if decode_max_len < 1 else decode_max_len
        max_new_tokens = max(1, max_new_tokens)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            do_sample=False,
            min_length=decode_min_len,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            length_penalty=llm_length_penalty,
            temperature=temperature,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )

        return generated_ids

    def _merge_input_ids_with_speech_features(
        self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None,
        speech_lens=None
    ):
        speech_lens = None
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.llm.config.pad_token_id))
        special_speech_token_mask = input_ids == self.llm.config.default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        max_embed_dim = (num_special_speech_tokens.max() * (speech_len - 1)) + sequence_length
        batch_indices, non_speech_indices = torch.where(input_ids != self.llm.config.default_speech_token_id)
        new_token_positions = (torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1)
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]
        final_embedding = torch.zeros(batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        final_attention_mask = torch.zeros(batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device)
        if labels is not None:
            final_labels = torch.full((batch_size, max_embed_dim), IGNORE_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)
        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (batch_indices.to(target_device), non_speech_indices.to(target_device), text_to_overwrite.to(target_device))
        attention_mask = attention_mask.to(target_device)
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_speech_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_speech_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_speech_indices]
        speech_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device)
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        if speech_lens is not None:
            speech_pad_position = speech_to_overwrite.cumsum(-1) <= speech_lens[:, None]
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[:, None].to(target_device)
        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation.")
        final_embedding[speech_to_overwrite] = (speech_features.contiguous().reshape(-1, embed_dim).to(target_device))
        if speech_lens is not None:
            speech_to_overwrite &= speech_pad_position
        final_attention_mask |= speech_to_overwrite
        batch_indices, pad_indices = torch.where(input_ids == self.llm.config.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]
        final_embedding[batch_indices, indices_to_mask] = 0
        if labels is None:
            final_labels = None
        return final_embedding, final_attention_mask, final_labels