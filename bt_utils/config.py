import numpy as np
import os
from .constants import *
import re


class Config:
    def __init__(self, filename=''):
        self.entries = []
        self.config_path = ''
        if filename:
            self.parse_file(filename)

    def config_file_path(self):
        return self.config_path

    def parse(self, content: str) -> None:
        # format: key = value\n
        # allow trailing # as comment
        pattern = r'^\s*(\S+)\s*?=\s*?(\S+)\s*?$'
        lines = content.split('\n')
        for line in lines:
            line = line.split('#', 1)[0]  # remove comments
            line = line.strip()
            if not line:
                continue
            match = re.match(pattern, line)
            if match:
                self.entries.append((match.group(1), match.group(2)))
            else:
                print(f'Invalid config line: {line}')
        # print("entries: ",self.entries)

    def parse_file(self, filename: str) -> None:
        try:
            with open(filename, 'r') as file:
                content = file.read()
                self.parse(content)
                self.config_path = filename
        except IOError:
            raise RuntimeError(f'Failed to open config file: {filename}')

    def to_string(self) -> str:
        result = ''
        for entry in self.entries:
            result += f'{entry[0]} = {entry[1]}\n'
        return result

    def __getitem__(self, key: str):
        for entry in self.entries:
            if entry[0] == key:
                return entry[1]
        raise KeyError(f'Config key not found: {key}')

    def __setitem__(self, key, value):
        for entry in self.entries:
            if entry[0] == key:
                entry[1] = value
                return
        self.entries.append((key, value))

    def has_key(self, key: str) -> str:
        for entry in self.entries:
            if entry[0] == key:
                return True
        return False

    def get(self, key: str, default=None) -> str:
        if self.has_key(key):
            return self[key]
        if default is not None:
            return default
        else:
            raise RuntimeError(f"expect {key} exists")

    def get_int(self, key: str, default=None) -> int:
        if self.has_key(key):
            return int(self[key])
        if default is not None:
            return default
        else:
            raise RuntimeError(f"expect {key} exists")

    def get_float(self, key: str, default=None) -> float:
        if self.has_key(key):
            return float(self[key])
        if default is not None:
            return default
        else:
            raise RuntimeError(f"expect {key} exists")

    def get_bool(self, key: str, default=None) -> bool:
        if self.has_key(key):
            return self[key].lower() in ['true', '1', 'yes', 'on']
        if default is not None:
            return default
        else:
            raise RuntimeError(f"expect {key} exists")


class InputData:
    def __init__(self):
        self.input_total_batch = 0
        self.max_seq_len = 0
        self.batch_seqlen = []
        self.all_data = []

    def from_file(self, input_file: str) -> None:
        with open(input_file, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                if line.strip().startswith("#"):
                    continue
                batch_data = [int(word) for word in line.split()]
                row = len(batch_data)
                self.all_data.append(batch_data)
                self.batch_seqlen.append(row)
                self.input_total_batch += 1
                self.max_seq_len = max(self.max_seq_len, row)

    def create_mock(self, batch: int, seq_len: int, vocab_size: int) -> None:
        self.input_total_batch = batch
        self.max_seq_len = seq_len
        for _ in range(batch):
            batch_data = [np.random.randint(vocab_size)
                          for _ in range(seq_len)]
            self.all_data.append(batch_data)
            self.batch_seqlen.append(seq_len)

    def get_padded_input(self, pad_id: int) -> None:
        padded_input = np.full(
            (self.input_total_batch, self.max_seq_len), pad_id)
        for i in range(self.input_total_batch):
            padded_input[i, :len(self.all_data[i])] = self.all_data[i]
        return padded_input


class GptConfig:
    def __init__(self):
        pass

    def is_glm(self):
        return 'glm' in self.model_name.lower() or 'glm2' in self.model_name.lower()

    def get_input_seqlen(self) -> int:
        return self.input_data_.max_seq_len

    def get_input_total_batch(self) -> int:
        return self.input_data_.input_total_batch

    def is_mock_input(self) -> bool:
        return self.use_mock_input_

    def is_mock_filter(self) -> bool:
        return self.use_mock_filter_

    def load_model_config(self, config: Config):
        self.model_config_ = config

        self.model_name = config.get("model_name")
        self.eos_id = config.get_int("eos_id")
        self.top_k = config.get_int("top_k")
        self.top_p = config.get_float("top_p")
        self.temperature = config.get_float("temperature")
        self.skip_embedding = config.get_bool("skip_embedding", False)
        self.embedding_mode = config.get_bool(
            "embedding_mode", "MAGICMIND_EMBEDDING")
        self.input_data_ = InputData()
        self.baseline_input_data = InputData()
        self.use_baseline = False
        self.beam_width = 1
        self.unfuse_residual_layernorm = False  # use individual bias-add-norm layer
        self.unfuse_gemm = False  # unfuse AttnProj and FeedForward, use gemms instead
        self.layernorm_eps = config.get_float("layernorm_eps",1e-5)
        self.input_size = config.get_int("input_size")
        self.tp_num = config.get_int("tp_num")
        self.layer_num = config.get_int("layer_num")
        self.beam_width = config.get_int("beam_width", 1)
        self.head_num = config.get_int("head_num")
        self.head_size = config.get_int("head_size")
        self.ffn_inner_size = config.get_int("ffn_inner_size")
        self.vocab_size = config.get_int("vocab_size")
        # unfuse_residual_layernorm not used for now
        # self.unfuse_residual_layernorm = config.get_bool("unfuse_residual_layernorm", False)
        self.unfuse_gemm = config.get_bool("unfuse_gemm", False)
        self.use_gptj_residual = config.get_bool("use_gptj_residual", False)
        self.use_gated_ffn = config.get_bool("use_gated_ffn", False)
        self.data_type = config.get("data_type", "CNNL_DTYPE_HALF")
        self.filter_data_type = config.get(
            "filter_data_type", "CNNL_DTYPE_HALF")
        self.context_mha_compute_data_type = config.get(
            "context_mha_compute_data_type", "CNNL_DTYPE_HALF")
        self.filter_bitwidth = config.get_int("filter_bitwidth", 0)
        self.quantize_layout = config.get(
            "quantize_layout", "CNNL_QUANTIZE_NONE")
        self.is_embedding_weight_quantized = config.get_bool(
            "is_embedding_weight_quantized", False)
        self.norm_type = config.get("norm_type", "CNNL_TRANSFORMER_LAYERNORM")
        self.lnres_mode = config.get(
            "lnres_mode", "CNNL_TRANSFORMER_PRE_LAYERNORM_INSIDE_RESIDUAL")
        self.residual_alpha = config.get_float("residual_alpha", 1.0)
        self.residual_beta = config.get_float("residual_beta", 1.0)
        self.act_mode = config.get("act_mode", "CNNL_ACTIVATION_GELU")
        self.mha_mul_factor_after_qk = config.get_bool(
            "mha_mul_factor_after_qk", False)
        self.position_embedding_type = config.get(
            "position_embedding_type", "CNNL_ATTN_FOLD_ROTARY_EMBEDDING")
        self.rotary_pos_emb_dim = config.get_int("rotary_pos_emb_dim", 0)
        self.preload_size = config.get_int("preload_size", 0)
        self.context_only_last_logit = config.get_bool(
            "context_only_last_logit", True)
        self.is_individual_token_offset = config.get_bool(
            "is_individual_token_offset", False)
        if is2DRotaryEmbedding(self.position_embedding_type):
            self.is_individual_token_offset = True
        if not isRotaryEmbedding(self.position_embedding_type) or self.rotary_pos_emb_dim <= 0:
            self.rotary_pos_emb_dim = self.head_size
        self.position_embedding_seq_len_q = config.get_int(
            "position_embedding_seq_len_q", 0)
        self.position_embedding_seq_len_k = config.get_int(
            "position_embedding_seq_len_k", 0)
        # self.cache_memory_len = config.get_int("cache_memory_len",0) # C++ demo没有去获取这个参数

    def __str__(self):
        params = [
            f"model_name: {self.model_name}",
            f"eos_id: {self.eos_id}",
            f"top_k: {self.top_k}",
            f"top_p: {self.top_p}",
            f"temperature: {self.temperature}",
            f"skip_embedding: {self.skip_embedding}",
            f"input_size: {self.input_size}",
            f"tp_num: {self.tp_num}",
            f"layer_num: {self.layer_num}",
            f"beam_width: {self.beam_width}",
            f"head_num: {self.head_num}",
            f"head_size: {self.head_size}",
            f"ffn_inner_size: {self.ffn_inner_size}",
            f"vocab_size: {self.vocab_size}",
            f"unfuse_residual_layernorm: {self.unfuse_residual_layernorm}",
            f"unfuse_gemm: {self.unfuse_gemm}",
            f"use_gptj_residual: {self.use_gptj_residual}",
            f"use_gated_ffn: {self.use_gated_ffn}",
            f"data_type: {self.data_type}",
            f"filter_data_type: {self.filter_data_type}",
            f"context_mha_compute_data_type: {self.context_mha_compute_data_type}",
            f"filter_bitwidth: {self.filter_bitwidth}",
            f"quantize_layout: {self.quantize_layout}",
            f"is_embedding_weight_quantized: {self.is_embedding_weight_quantized}",
            f"norm_type: {self.norm_type}",
            f"lnres_mode: {self.lnres_mode}",
            f"residual_alpha: {self.residual_alpha}",
            f"residual_beta: {self.residual_beta}",
            f"act_mode: {self.act_mode}",
            f"mha_mul_factor_after_qk: {self.mha_mul_factor_after_qk}",
            f"position_embedding_type: {self.position_embedding_type}",
            f"rotary_pos_emb_dim: {self.rotary_pos_emb_dim}",
            f"cache_memory_len: {self.cache_memory_len}",
            f"preload_size: {self.preload_size}",
            f"position_embedding_seq_len_q: {self.position_embedding_seq_len_q}",
            f"position_embedding_seq_len_k: {self.position_embedding_seq_len_k}"
        ]
        return "\n".join(params)

    def enumerate_dir(self, directory):
        ret = []
        for entry in os.scandir(directory):
            if entry.is_file():
                ret.append(os.path.join(directory, entry.name))
        return ret

    def load_input_config(self, config: Config):
        config_dir = "./"
        config_file = config.config_file_path()
        if "/" in config_file:
            config_dir = config_file[:config_file.rfind("/") + 1]
        if config.has_key("baseline_tokens"):
            self.baseline_token_path = config.get("baseline_tokens")
            if self.baseline_token_path[0] != "/":
                self.baseline_token_path = config_dir + self.baseline_token_path
            self.baseline_input_data.from_file(self.baseline_token_path)
            self.use_baseline = True
        if not config.has_key("input_path") or config.get("input_path") == "":
            print("input_path not set, use mock input")
            self.use_mock_input_ = True
            if config.has_key("mock_input_batch_size") and config.has_key("mock_input_seq_len"):
                self.mock_input_batch_size = config.get_int(
                    "mock_input_batch_size")
                self.mock_input_seq_len = config.get_int("mock_input_seq_len")
            else:
                raise Exception(
                    "mock_input_batch_size and mock_input_seq_len must be set when input_path is not set")
            self.input_data_.create_mock(
                self.mock_input_batch_size, self.mock_input_seq_len, self.model_config_.get_int("vocab_size"))
        else:
            self.input_path = config.get("input_path")
            if self.input_path[0] != "/":
                self.input_path = config_dir + self.input_path
            self.use_mock_input_ = False
            self.input_data_.from_file(self.input_path)
            if config.has_key("mock_input_batch_size") or config.has_key("mock_input_seq_len"):
                print(
                    "mock_input_batch_size and mock_input_seq_len will be ignored when input_path is set")
            for i in range(self.input_data_.input_total_batch):
                for j in range(len(self.input_data_.all_data[i])):
                    word = self.input_data_.all_data[i][j]
                    if word < 0:
                        raise ValueError(
                            f"input data batch {i} token {j} is negative: {word}")
                    if word > self.vocab_size:
                        raise ValueError(
                            f"input data batch {i} token {j} is out of vocab: {word}, vocab_size: {self.vocab_size}")
        # if not config.has_key("filter_path") or (self.filter_path := ) == "":
        if not config.has_key("filter_path") or config.get("filter_path") == "":
            print("filter_path not set, use mock filter")
            self.use_mock_filter_ = True
            self.filter_path = None
        else:
            self.filter_path = config.get("filter_path")
            if self.filter_path[0] != "/":
                self.filter_path = config_dir + self.filter_path
            self.use_mock_filter_ = False
            self.filter_files = self.enumerate_dir(self.filter_path)

        self.once_batch_size = config.get_int("once_batch_size", 1)
        self.output_size = config.get_int("output_size")
        self.position_embedding_seq_len_q = config.get_int(
            "position_embedding_seq_len_q", 0)
        self.position_embedding_seq_len_k = config.get_int(
            "position_embedding_seq_len_k", 0)
        self.cache_memory_len = config.get_int(
            "cache_memory_len", 0)

        if self.once_batch_size <= 0:
            print("once_batch_size not set, infer from input data")
            self.once_batch_size = self.input_data_.input_total_batch

        if self.output_size <= 0:
            print("output_size not set, assume same as input_seq_len")
            self.output_size = self.input_data_.max_seq_len

        if not self.cache_memory_len or self.cache_memory_len <= 0:
            print("cache_memory_len not set, assume input_seq_len + output_size + 1")
            self.cache_memory_len = self.input_data_.max_seq_len + self.output_size + 1

        if not self.position_embedding_seq_len_q or not self.position_embedding_seq_len_k \
                or self.position_embedding_seq_len_q <= 0 or self.position_embedding_seq_len_k <= 0:
            print(
                "position_embedding_seq_len_q/k not set, assume same as cache_memory_len")

        self.once_batch_size = min(
            self.once_batch_size, self.input_data_.input_total_batch)
        self.cache_memory_len = max(
            self.cache_memory_len, self.input_data_.max_seq_len + self.output_size + 1)
        self.position_embedding_seq_len_q = max(
            self.cache_memory_len, self.position_embedding_seq_len_q)
        self.position_embedding_seq_len_k = max(
            self.cache_memory_len, self.position_embedding_seq_len_k)

    def get_input_data(self) -> InputData:
        return self.input_data_

    def get_input_seqLen(self) -> int:
        return self.input_data_.max_seq_len

    def get_input_total_batch(self) -> int:
        return self.input_data_.input_total_batch
