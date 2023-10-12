import argparse
import torch
import torch_mlu
import time

from quantization import quantize

from SwissArmyTransformer import get_args, get_tokenizer
from SwissArmyTransformer.arguments import initialize_distributed
from SwissArmyTransformer.training import load_checkpoint
from SwissArmyTransformer.model import GLM130B
from SwissArmyTransformer.mpu import get_model_parallel_world_size, get_model_parallel_rank, get_model_parallel_group
from bt_utils.utils import *
from bt_utils.config import *
from bt_utils import comm
from bt_utils.gptweights import GPTWeights
from bt_utils.gpt import GPT
import torch.nn as nn

def add_bminf_args(parser):
    """Arguments for BMInf"""
    group = parser.add_argument_group("BMInf")

    group.add_argument("--bminf", action="store_true", help="Use BMInf to support low resource evaluation")
    group.add_argument("--bminf-memory-limit", type=int, default=20, help="Max memory for model per GPU (in GB)")
    return parser


def add_quantization_args(parser):
    group = parser.add_argument_group("Quantization")

    group.add_argument("--quantization-bit-width", type=int, default=None)
    group.add_argument("--from-quantized-checkpoint", action="store_true", help="Loading from a quantized checkpoint")


def add_initialization_args(parser):
    group = parser.add_argument_group("Initialization")

    group.add_argument(
        "--sequential-initialization",
        action="store_true",
        help="Initialize sequentially in tensor parallel group (reduce CPU RAM for initialization)",
    )


def initialize(extra_args_provider):
    parser = argparse.ArgumentParser(add_help=False)
    add_bminf_args(parser)
    add_quantization_args(parser)
    add_initialization_args(parser)
    GLM130B.add_model_specific_args(parser)
    extra_args_provider(parser)
    known, args_list = parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.do_train = False
    initialize_distributed(args)
    return args

class BTModel(nn.Module):
    def __init__(
        self,
        embedding_layer,
        context_decoder,
        device
    ) -> None:
        super(BTModel, self).__init__()
        self._embedding_layer = embedding_layer
        self._context_decoder = context_decoder
        self.device = device
    
    def forward(
        self,
        input_ids,
        token_offset,
        mask,
        valid_token,
        k_cache,
        v_cache,
        curr_idx
    ):
        embedding_output = self._embedding_layer(input_ids)
        output, _, _ = self._context_decoder.forward(
			embedding_output,
			token_offset,
			mask,
			valid_token,
			None,
			None,
			None)
        return output


def initialize_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)

    gpt_config = GptConfig()
    gpt_config.load_model_config(Config("./chatglm_130B_int8.cfg"))
    gpt_config.load_input_config(Config("./input_int8.cfg"))
    torch.mlu.set_device(torch.distributed.get_rank())
    device = torch.mlu.current_device()

    embedding_path = gpt_config.filter_path + "/" + gpt_config.model_name + ".embed_in.weight.bin"

    input_data = gpt_config.get_input_data()
    padded_input = input_data.get_padded_input(gpt_config.eos_id)
    max_cache_len = input_data.max_seq_len + gpt_config.output_size + 1 # FIXME:这个变量好像没用
    rank = torch.distributed.get_rank()

    embedding_filter = get_embedding_filters(gpt_config)
    quantize_bitwidth = gpt_config.filter_bitwidth
    if quantize_bitwidth != 0:
        print("Model is quantized with bitwidth =", quantize_bitwidth)
        if gpt_config.filter_data_type != "CNNL_DTYPE_INT8":
            raise ValueError("Model is quantized with bitwidth = " + str(quantize_bitwidth) + ", but filter data type is not CNNL_DTYPE_INT8.")

    embedding_layer = build_embedding(gpt_config.vocab_size,gpt_config.input_size,embedding_filter)
    embedding_layer.half().to(device)

    weight = GPTWeights(
            model_name=gpt_config.model_name,
            tp_num=gpt_config.tp_num,
            head_num=gpt_config.head_num,
            head_size=gpt_config.head_size,
            layer_num=gpt_config.layer_num,
            ffn_inner_size=gpt_config.ffn_inner_size,
            vocab_size=gpt_config.vocab_size,
            filter_bitwidth=gpt_config.filter_bitwidth,
            filter_data_type=gpt_config.filter_data_type,
            data_type=gpt_config.data_type,
            position_embedding_seq_len_q=gpt_config.position_embedding_seq_len_q,
            position_embedding_seq_len_k=gpt_config.position_embedding_seq_len_k,
            rotary_pos_emb_dim=gpt_config.rotary_pos_emb_dim,
            input_size=gpt_config.input_size,
            rank=rank,
            device=device,
            position_embedding_type=gpt_config.position_embedding_type,
            is_embedding_weight_quantized=gpt_config.is_embedding_weight_quantized,
            filter_path=gpt_config.filter_path,
            is_mock_filter=gpt_config.is_mock_filter(),
            use_gated_ffn=gpt_config.use_gated_ffn,
            is_binary_file=True)

    decoder_args = {
        "model_name":gpt_config.model_name,
        "tp_num":gpt_config.tp_num,
        "layer_num":gpt_config.layer_num,
        "beam_width":gpt_config.beam_width,
        "input_size":gpt_config.input_size,
        "head_num":gpt_config.head_num,
        "head_size":gpt_config.head_size,
        "ffn_inner_size":gpt_config.ffn_inner_size,
        "vocab_size":gpt_config.vocab_size,
        "unfuse_residual_layernorm":gpt_config.unfuse_residual_layernorm,
        "unfuse_gemm":gpt_config.unfuse_gemm,
        "use_gptj_residual":gpt_config.use_gptj_residual,
        "use_gated_ffn":gpt_config.use_gated_ffn,
        "data_type":gpt_config.data_type,
        "filter_data_type":gpt_config.filter_data_type,
        "context_mha_compute_data_type":gpt_config.context_mha_compute_data_type,
        "filter_bitwidth":gpt_config.filter_bitwidth,
        "quantize_layout":gpt_config.quantize_layout,
        "norm_type":gpt_config.norm_type,
        "lnres_mode":gpt_config.lnres_mode,
        "residual_alpha":gpt_config.residual_alpha,
        "layernorm_eps": gpt_config.layernorm_eps,
        "residual_beta":gpt_config.residual_beta,
        "act_mode":gpt_config.act_mode,
        "position_embedding_type":gpt_config.position_embedding_type,
        "rotary_pos_emb_dim":gpt_config.rotary_pos_emb_dim,
        "position_embedding_seq_len_q":gpt_config.position_embedding_seq_len_q,
        "position_embedding_seq_len_k":gpt_config.position_embedding_seq_len_k, 
        "cache_memory_len":gpt_config.cache_memory_len,
        "context_only_last_logit":gpt_config.context_only_last_logit,
        "weight":weight,
        "preload_size":gpt_config.preload_size,
        "mha_mul_factor_after_qk":gpt_config.mha_mul_factor_after_qk,
        "is_individual_token_offset":gpt_config.is_individual_token_offset,
        "is_embedding_weight_quantized":gpt_config.is_embedding_weight_quantized,
        "is_context_decoder":True,
    }

    torch.classes.load_library("/workspace/bangtransformer/build/libbttorch.so")
    context_decoder = GPT(
        **decoder_args
    )
    context_decoder.set_context_max_batch_and_seq_len(1, 2048)
    model = BTModel(embedding_layer, context_decoder, device)

    return model, tokenizer
