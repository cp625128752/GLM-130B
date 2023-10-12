import torch
import typing
import torch_mlu
import torch.distributed as dist
from torch import nn
from .gptweights import GPTWeights
from .constants import *


class GPT(nn.Module):
    def __init__(self,
                 unfuse_residual_layernorm: bool,
                 unfuse_gemm: bool,
                 use_gptj_residual: bool,
                 context_mha_compute_data_type: str,
                 quantize_layout: str,
                 norm_type: str,
                 lnres_mode: str,
                 residual_alpha: float,
                 residual_beta: float,
                 layernorm_eps: float,
                 act_mode: str,
                 cache_memory_len: int,
                 is_context_decoder: bool,
                 model_name: str,
                 tp_num: int,
                 head_num: int,
                 head_size: int,
                 layer_num: int,
                 ffn_inner_size: int,
                 vocab_size: int,
                 filter_bitwidth: int,
                 filter_data_type: str,
                 data_type: str,
                 position_embedding_seq_len_q: int,
                 position_embedding_seq_len_k: int,
                 rotary_pos_emb_dim: int,
                 input_size: int,
                 position_embedding_type: str,
                 weight: GPTWeights,
                 is_embedding_weight_quantized: bool,
                 preload_size: int,
                 context_only_last_logit: bool,
                 is_individual_token_offset: bool,
                 mha_mul_factor_after_qk: bool,
                 use_gated_ffn: bool = False,
                 beam_width: int = 1
                 ):
        if filter_bitwidth != 0:
            print("Model is quantized with bitwidth =", filter_bitwidth)
            if filter_data_type != "CNNL_DTYPE_INT8":
                raise ValueError("Model is quantized with bitwidth = " + str(
                    filter_data_type) + ", but filter data type is not CNNL_DTYPE_INT8.")

        assert model_name is not None, "GPTWeights'sself.model_name must not be None"
        assert tp_num > 0
        assert layer_num > 0
        assert head_num > 0
        assert head_size > 0

        assert filter_bitwidth >= 0
        assert position_embedding_seq_len_q > 0
        assert position_embedding_seq_len_k > 0
        assert rotary_pos_emb_dim > 0
        assert input_size > 0

        assert ffn_inner_size > 0
        assert vocab_size > 0

        assert context_mha_compute_data_type in dataTypes
        assert quantize_layout in QuantLayouts
        assert norm_type in normTypes
        assert lnres_mode in LayernormResidualStructures
        assert act_mode in Acts
        assert filter_data_type in dataTypes
        assert data_type in dataTypes
        assert position_embedding_type in posEmbedTypes

        self.is_context_decoder = is_context_decoder
        self.weight = weight
        self.gen_iter = 0
        self.cache_memory_len = cache_memory_len

        args = [
            tp_num,
            layer_num,
            beam_width,
            input_size,
            head_num,
            head_size,
            ffn_inner_size,
            vocab_size,
            unfuse_residual_layernorm,
            unfuse_gemm,
            use_gptj_residual,
            use_gated_ffn,
            data_type,
            filter_data_type,
            context_mha_compute_data_type,
            filter_bitwidth,
            quantize_layout,
            norm_type,
            lnres_mode,
            residual_alpha,
            residual_beta,
            layernorm_eps,
            act_mode,
            position_embedding_type,
            rotary_pos_emb_dim,
            position_embedding_seq_len_q,
            position_embedding_seq_len_k,
            cache_memory_len,
            self.weight.layer_filters,
            self.weight.logit_layernorm_scale,
            self.weight.logit_layernorm_bias,
            self.weight.logit_filter,
            self.weight.logit_bias,
            self.weight.position_embedding,
            self.weight.zeros,
            self.weight.per_channel_scale,
            preload_size,
            torch.mlu.current_stream().mlu_stream,
            is_embedding_weight_quantized,
            mha_mul_factor_after_qk,
            is_individual_token_offset,
            context_only_last_logit,
            is_context_decoder,
        ]

        self.decoder = torch.classes.bangnsformer.GptDecoderOp()
        self.decoder.setup(*args)

    def set_context_max_batch_and_seq_len(self, max_batch_size:int,
                                            context_max_seq_len:int):
        assert max_batch_size > 0,"max_batch_size should > 0"
        assert context_max_seq_len > 0,"context_max_seq_len should > 0"
        assert self.is_context_decoder, "only context decoder should call set_max_batch_and_context_seqlen"
        self.max_batch_size = max_batch_size
        self.context_max_seq_len = context_max_seq_len
        self.decoder.set_context_max_batch_and_seq_len(max_batch_size,context_max_seq_len)

    def set_generate_max_batch(self, max_batch_size:int):
        assert max_batch_size > 0,"max_batch_size should > 0"
        assert not self.is_context_decoder, "only generate decoder should call set_generate_max_batch"
        self.max_batch_size = max_batch_size
        self.decoder.set_generate_max_batch(max_batch_size)

    def forward(
        self,
        input: torch.Tensor,
        token_offset: torch.IntTensor,
        mask: typing.Optional[torch.IntTensor],
        valid_token: typing.Optional[torch.IntTensor],
        k_cache: typing.Optional[torch.Tensor],
        v_cache: typing.Optional[torch.Tensor],
        curr_idx: typing.Optional[torch.IntTensor],
    ):
        if self.is_context_decoder:
            logits, k_cache, v_cache = self.decoder.forward(
                input,
                token_offset,
                mask,
                valid_token,
                None,  # k_cache,
                None,  # v_cache
                None,  # curr_idx
            )
            return logits, k_cache, v_cache
        else:
            logits, k_cache, v_cache = self.decoder.forward(
                input,
                token_offset,
                None,  # mask
                None,  # valid_token
                k_cache,
                v_cache,
                curr_idx)
            self.gen_iter +=1
        return logits, k_cache, v_cache