from typing import Dict, List, Tuple
import operator
from functools import reduce
import torch
import torch_mlu
from .constants import *
import os


class GPTWeights:
    def __init__(self,
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
                 rank: int,
                 device,
                 position_embedding_type: str,
                 is_embedding_weight_quantized: bool,
                 filter_path: str = None,
                 is_mock_filter: bool = False,
                 use_gated_ffn: bool = False,  # FIXME: 应该是默认False吗
                 is_binary_file: bool = True,  # FIXME: 应该是默认False吗
                 ) -> None:

        # hidden_size = self.head_num * self.head_size
        assert model_name is not None, "GPTWeights'sself.model_name must not be None"
        assert (filter_path is not None or is_mock_filter is True)
        assert tp_num > 0
        assert layer_num > 0
        assert head_num > 0
        assert head_size > 0

        assert filter_bitwidth >= 0
        assert position_embedding_seq_len_q > 0
        assert position_embedding_seq_len_k > 0
        assert rotary_pos_emb_dim > 0
        assert input_size > 0
        assert device is not None
        assert rank >= 0

        assert ffn_inner_size > 0
        assert vocab_size > 0
        assert is_binary_file == True
        if not is_mock_filter:
            assert os.path.exists(filter_path)
        
        assert filter_data_type in dataTypes, f'expect filter_data_type in {dataTypes}, but got {filter_data_type}'
        assert data_type in dataTypes, f'expect data_type in {dataTypes}, but got {data_type}'
        assert position_embedding_type in posEmbedTypes

        # filter_data_type =  dataTypeMap[filter_data_type]
        # data_type =  dataTypeMap[data_type]
        # position_embedding_type =  posEmbedTypeMap[position_embedding_type]


        self.device = device
        self.filter_bitwidth = filter_bitwidth
        self.filter_data_type = filter_data_type
        self.data_type = data_type
        self.position_embedding_seq_len_q = position_embedding_seq_len_q
        self.position_embedding_seq_len_k = position_embedding_seq_len_k
        self.rotary_pos_emb_dim = rotary_pos_emb_dim
        self.position_embedding_type = position_embedding_type
        self.input_size = input_size
        self.rank = rank
        self.is_embedding_weight_quantized = is_embedding_weight_quantized
        assert is_embedding_weight_quantized is False
        self.head_num = head_num
        self.head_size = head_size
        self.tp_num = tp_num
        self.ffn_inner_size = ffn_inner_size
        self.vocab_size = vocab_size
        self.layer_num = layer_num
        self.model_name = model_name
        self.filter_path = filter_path
        self.is_mock_filter = is_mock_filter
        self.use_gated_ffn = use_gated_ffn
        self.filter_status = {
            "self_attn_q_filter": {"is_tp": True, "is_per_layer": True},
            "self_attn_k_filter": {"is_tp": True, "is_per_layer": True},
            "self_attn_v_filter": {"is_tp": True, "is_per_layer": True},
            "self_attn_out_filter": {"is_tp": True, "is_per_layer": True},
            "self_attn_q_bias": {"is_tp": True, "is_per_layer": True},
            "self_attn_k_bias": {"is_tp": True, "is_per_layer": True},
            "self_attn_v_bias": {"is_tp": True, "is_per_layer": True},
            "ffn_inner_filter": {"is_tp": True, "is_per_layer": True},
            "ffn_gate_filter": {"is_tp": True, "is_per_layer": True},
            "ffn_out_filter": {"is_tp": True, "is_per_layer": True},
            "ffn_inner_bias": {"is_tp": True, "is_per_layer": True},
            "ffn_gate_bias": {"is_tp": True, "is_per_layer": True},
            "self_attn_layer_norm_scale": {"is_tp": False, "is_per_layer": True},
            "self_attn_layer_norm_bias": {"is_tp": False, "is_per_layer": True},
            "ffn_layer_norm_scale": {"is_tp": False, "is_per_layer": True},
            "ffn_layer_norm_bias": {"is_tp": False, "is_per_layer": True},
            "self_attn_out_bias": {"is_tp": False, "is_per_layer": True},
            "ffn_out_bias": {"is_tp": False, "is_per_layer": True},
            "logit_filter": {"is_tp": True, "is_per_layer": False},
            "logit_bias": {"is_tp": True, "is_per_layer": False},
            "position_embedding": {"is_tp": True, "is_per_layer": False},
            "logit_layer_norm_scale": {"is_tp": False, "is_per_layer": False},
            "logit_layer_norm_bias": {"is_tp": False, "is_per_layer": False}
        }

        self.filter_map = self.get_filter_path_map()
        if filter_bitwidth != 0:
            self.per_channel_scale = self.load_quantize_scales()
        else:
            self.per_channel_scale = {}
        self.layer_filters, self.logit_layernorm_scale, self.logit_layernorm_bias, self.logit_filter, self.logit_bias, self.position_embedding, self.zeros = self.load_single_gpt_tp_filters_from_file(
            self.filter_map, is_binary_file=is_binary_file)

    def get_path_1(self, type):
        out = []
        if self.is_mock_filter:
            out.append("mock_fn")
        else:
            file_path = os.path.join(
                self.filter_path, self.model_name + "." + type + ".bin")
            out.append(file_path)
        return out

    def get_path_2(self, num, id, type):
        out = []
        for i in range(num):
            if self.is_mock_filter:
                out.append("mock_fn")
            else:
                file_path = os.path.join(
                    self.filter_path, self.model_name + "." + id + "." + str(i) + "." + type + ".bin")
                out.append(file_path)
        return out

    def get_path_3(self, id2, type):
        num1, id1, num2, = self.tp_num, "tp", self.layer_num
        out = []
        for i in range(num1):
            for j in range(num2):
                if self.is_mock_filter:
                    out.append("mock_fn")
                else:
                    file_path = os.path.join(self.filter_path, self.model_name + "." + id1 + "." + str(
                        i) + "." + id2 + "." + str(j) + "." + type + ".bin")
                    out.append(file_path)
        return out

    def get_filter_path_map(self):
        filter_path_map = {}
        attn_q_filter_path = self.get_path_3( "layers", "attention.query.weight")
        attn_k_filter_path = self.get_path_3( "layers", "attention.key.weight")
        attn_v_filter_path = self.get_path_3( "layers", "attention.value.weight")
        attn_out_filter_path = self.get_path_3( "layers", "attention.dense.weight")
        ffn_inner_filter_path = self.get_path_3( "layers", "mlp.dense_h_to_4h.weight")
        ffn_outer_filter_path = self.get_path_3( "layers", "mlp.dense_4h_to_h.weight")
        logit_filter_path = self.get_path_2( self.tp_num, "tp", "embed_out.weight")
        attn_ln_scale_path = self.get_path_2( self.layer_num, "layers", "input_layernorm.weight")
        attn_ln_bias_path = self.get_path_2( self.layer_num, "layers", "input_layernorm.bias")
        attn_q_bias_path = self.get_path_3( "layers", "attention.query.bias")
        attn_k_bias_path = self.get_path_3( "layers", "attention.key.bias")
        attn_v_bias_path = self.get_path_3( "layers", "attention.value.bias")
        attn_out_bias_path = self.get_path_2( self.layer_num, "layers", "attention.dense.bias")
        ffn_ln_scale_path = self.get_path_2( self.layer_num, "layers", "post_attention_layernorm.weight")
        ffn_ln_bias_path = self.get_path_2( self.layer_num, "layers", "post_attention_layernorm.bias")
        ffn_inner_bias_path = self.get_path_3( "layers", "mlp.dense_h_to_4h.bias")
        ffn_outer_bias_path = self.get_path_2( self.layer_num, "layers", "mlp.dense_4h_to_h.bias")
        logit_ln_scale_path = self.get_path_1( "final_layer_norm.weight")
        logit_ln_bias_path = self.get_path_1( "final_layer_norm.bias")

        position_embedding_path = []
        if self.position_embedding_type == "CNNL_ATTN_RELATIVE_POSITION_EMBEDDING":
            position_embedding_path = self.get_path_3(
                "layers", "rpe_pos_emb", )
        elif self.position_embedding_type in ["CNNL_ATTN_CROSS_ROTARY_EMBEDDING",
                                              "CNNL_ATTN_CROSS_ROTARY_EMBEDDING_2D",
                                              "CNNL_ATTN_FOLD_ROTARY_EMBEDDING",
                                              "CNNL_ATTN_FOLD_ROTARY_EMBEDDING_2D"]:
            position_embedding_path = self.get_path_2(
                self.tp_num, "tp", "rotary_pos_emb")
        elif self.position_embedding_type == "CNNL_ATTN_ALIBI_POSITION_EMBEDDING":
            position_embedding_path = self.get_path_2(
                self.tp_num, "tp", "alibi_pos_emb")
        else:
            raise RuntimeError(f"{self.position_embedding_type} is invalid")

        if self.use_gated_ffn:
            ffn_gate_filter_path = self.get_path_3(
                "layers", "mlp.w3.weight")
            ffn_gate_bias_path = self.get_path_3(
                "layers", "mlp.w3.bias")

        filter_path_map["self_attn_q_filter"] = attn_q_filter_path
        filter_path_map["self_attn_k_filter"] = attn_k_filter_path
        filter_path_map["self_attn_v_filter"] = attn_v_filter_path
        filter_path_map["self_attn_out_filter"] = attn_out_filter_path
        filter_path_map["ffn_inner_filter"] = ffn_inner_filter_path
        filter_path_map["ffn_out_filter"] = ffn_outer_filter_path
        filter_path_map["logit_filter"] = logit_filter_path
        filter_path_map["self_attn_layer_norm_scale"] = attn_ln_scale_path
        filter_path_map["self_attn_layer_norm_bias"] = attn_ln_bias_path
        filter_path_map["self_attn_q_bias"] = attn_q_bias_path
        filter_path_map["self_attn_k_bias"] = attn_k_bias_path
        filter_path_map["self_attn_v_bias"] = attn_v_bias_path
        filter_path_map["self_attn_out_bias"] = attn_out_bias_path
        filter_path_map["ffn_layer_norm_scale"] = ffn_ln_scale_path
        filter_path_map["ffn_layer_norm_bias"] = ffn_ln_bias_path
        filter_path_map["ffn_inner_bias"] = ffn_inner_bias_path
        filter_path_map["ffn_out_bias"] = ffn_outer_bias_path
        filter_path_map["logit_layer_norm_scale"] = logit_ln_scale_path
        filter_path_map["logit_layer_norm_bias"] = logit_ln_bias_path
        filter_path_map["position_embedding"] = position_embedding_path
        if self.use_gated_ffn:
            filter_path_map["ffn_gate_filter"] = ffn_gate_filter_path
            filter_path_map["ffn_gate_bias"] = ffn_gate_bias_path
        return filter_path_map

    def load_quantize_scales(
        self
    ) -> Dict[str, List[torch.Tensor]]:
        ret = {
            "self_attn_q_filter_scale": [],
            "self_attn_k_filter_scale": [],
            "self_attn_v_filter_scale": [],
            "self_attn_out_filter_scale": [],
            "ffn_inner_filter_scale": [],
            "ffn_out_filter_scale": [],
            "ffn_gate_filter_scale": [],
            "logit_filter_scale": [],
        }
        hidden_size = self.head_num * self.head_size
        # FIXME: 仅仅为了glm 130B吗
        assert hidden_size % self.tp_num == 0
        tp_hs = int(hidden_size / self.tp_num)
        for i in range(self.layer_num):
            ret["self_attn_q_filter_scale"].append(
                self.readLayerOpScale(
                    "attention.query.weight_scale", i, tp_hs)
            )
            ret["self_attn_k_filter_scale"].append(
                self.readLayerOpScale(
                    "attention.key.weight_scale", i, tp_hs)
            )
            ret["self_attn_v_filter_scale"].append(
                self.readLayerOpScale(
                    "attention.value.weight_scale", i, tp_hs)
            )
            ret["self_attn_out_filter_scale"].append(
                self.readLayerOpScale(
                    "attention.dense.weight_scale", i, hidden_size)
            )
            ret["ffn_inner_filter_scale"].append(
                self.readLayerOpScale(
                    "mlp.dense_h_to_4h.weight_scale", i, self.ffn_inner_size)
            )
            ret["ffn_out_filter_scale"].append(
                self.readLayerOpScale(
                    "mlp.dense_4h_to_h.weight_scale", i, hidden_size)
            )
            if self.use_gated_ffn:
                ret["ffn_gate_filter_scale"].append(
                    self.readLayerOpScale(
                        "mlp.w3.weight_scale", i, self.ffn_inner_size)
                )
        ret["logit_filter_scale"].append(
            self.readOpScale("embed_out.weight", self.vocab_size))
        for k,v in ret.items():
            for idx, item in enumerate(v):
                assert item.dtype == torch.float32 , f"{k}'s {idx} dtype is {item.dtype}"
        return ret

    def readLayerOpScale(self, op_name: str, layer: int, count: int) -> np.ndarray:
        if self.is_mock_filter:
            return torch.zeros(count,dtype=torch.float32)
        path = os.path.join(self.filter_path,f"{self.model_name}.tp.{self.rank}.layers.{layer}.{op_name}.bin")
        print("path: ", path)
        assert os.path.exists(path), f"{path} not exists "
        return self.readBin(path, count)

    def readOpScale(self, op_name: str, count: int) -> np.ndarray:
        if self.is_mock_filter:
            return torch.zeros(count,dtype=torch.float32)
        path = os.path.join(self.filter_path,f"{self.model_name}.tp.{self.rank}.{op_name}.bin")
        assert os.path.exists(path), f"{path} not exists "
        return self.readBin(path, count)

    def readBin(self,path: str, count: int) -> np.ndarray:
        scale_arr = torch.zeros(count, dtype=torch.float32)
        with open(path, "rb") as reader:
            scale_arr = torch.Tensor(np.frombuffer(reader.read(count * 4), dtype=np.float32))
        return scale_arr

    def from_file(self, filter_file, meta: Tuple):
        dtype = meta[0]
        shape = meta[1]
        dtype = dataTypeMap[dtype]
        if self.is_mock_filter:
            return torch.tensor(np.empty(shape,dtype=dtype))
        with open(filter_file, 'rb') as file:
            filter_data = np.frombuffer(file.read(), dtype=dtype)
            filter_data = torch.tensor(filter_data)
            if filter_file.endswith("rotary_pos_emb.bin"):
                filter_data = filter_data.flatten()[:reduce(
                    operator.mul, shape)].reshape(shape)
            else:
                filter_data = filter_data.reshape(shape)
            return filter_data

    def gpt_decoder_filter_from_file(self, filter, is_binary_file):
        device = self.device
        filter_bitwidth = self.filter_bitwidth
        filter_data_type = self.filter_data_type
        data_type = self.data_type
        position_embedding_seq_len_q = self.position_embedding_seq_len_q
        position_embedding_seq_len_k = self.position_embedding_seq_len_k
        rotary_pos_emb_dim = self.rotary_pos_emb_dim
        position_embedding_type = self.position_embedding_type
        input_size = self.input_size

        assert is_binary_file
        seg_head_num = int(self.head_num/self.tp_num)
        seg_ffn_inner_size = int(self.ffn_inner_size / self.tp_num)
        seg_vocab_size = int(self.vocab_size / self.tp_num)
        hidden_size = seg_head_num * self.head_size
        zero_size = max(self.input_size, hidden_size)
        zeros = torch.zeros((zero_size,), dtype=torch.float32).to(device)
        shape_pe = None
        position_embedding = None

        print(f"position_embedding_type is : {position_embedding_type}")
        if position_embedding_type in ["CNNL_ATTN_CROSS_ROTARY_EMBEDDING",
                                       "CNNL_ATTN_FOLD_ROTARY_EMBEDDING"]:
            shape_pe = (position_embedding_seq_len_q, rotary_pos_emb_dim * 2)
        elif position_embedding_type in ["CNNL_ATTN_CROSS_ROTARY_EMBEDDING_2D",
                                         "CNNL_ATTN_FOLD_ROTARY_EMBEDDING_2D"]:
            shape_pe = (position_embedding_seq_len_q, rotary_pos_emb_dim)
        elif position_embedding_type in ["CNNL_ATTN_RELATIVE_POSITION_EMBEDDING",
                                         "CNNL_ATTN_ALIBI_POSITION_EMBEDDING"]:
            shape_pe = (seg_head_num, position_embedding_seq_len_q,
                        position_embedding_seq_len_k)
        elif position_embedding_type == "CNNL_ATTN_NO_POSITION_EMBEDDING":
            shape_pe = ()
        else:
            raise RuntimeError(
                f"position_embedding_type is out of range, position_embedding_type : {position_embedding_type}")

        method = self.from_file if is_binary_file else None
        if shape_pe:
            position_embedding_meta = (data_type, shape_pe)
            position_embedding = method(
                filter["position_embedding"][0], position_embedding_meta).to(device)

        filter_ci_div = 8 // filter_bitwidth if filter_data_type == "CNNL_DTYPE_INT8" else 1
        qkv_filter_meta = (filter_data_type, (hidden_size,
                           input_size // filter_ci_div))
        qkv_bias_meta = (data_type, (hidden_size,))
        input_bias_meta = (data_type, (input_size,))
        output_filter_meta = (
            filter_data_type, (input_size, hidden_size // filter_ci_div))
        ffn1_meta = (filter_data_type, (seg_ffn_inner_size,
                     input_size // filter_ci_div))
        ffn1_bias_meta = (data_type, (seg_ffn_inner_size,))
        ffn2_meta = (filter_data_type, (input_size,
                     seg_ffn_inner_size // filter_ci_div))

        layer_filters = []
        for i in range(self.layer_num):
            layer_filter_meta = {
                "filter_q": qkv_filter_meta,
                "filter_k": qkv_filter_meta,
                "filter_v": qkv_filter_meta,
                "bias_q": qkv_bias_meta,
                "bias_k": qkv_bias_meta,
                "bias_v": qkv_bias_meta,
                "attn_layernorm_scale": input_bias_meta,
                "attn_layernorm_bias": input_bias_meta,
                "filter_out": output_filter_meta,
                "bias_out": input_bias_meta,
                "ffn_filter1": ffn1_meta,
                "ffn_bias1": ffn1_bias_meta,
                "ffn_filter2": ffn2_meta,
                "ffn_bias2": input_bias_meta,
                "ffn_filter3": ffn1_meta,
                "ffn_bias3": ffn1_bias_meta,
                "ffn_layernorm_scale": input_bias_meta,
                "ffn_layernorm_bias": input_bias_meta,
                "zeros": zeros,
            }
            layer_filter = {
                "ffn_filter3": torch.tensor([]).to(torch.half).to(device),
                "ffn_bias3": torch.tensor([]).to(torch.half).to(device),

            }
            print(f"layer {i}..... ")
            layer_filter["filter_q"] = method(
                filter["self_attn_q_filter"][i], layer_filter_meta['filter_q']).to(device)
            layer_filter["filter_k"] = method(
                filter["self_attn_k_filter"][i], layer_filter_meta['filter_k']).to(device)
            layer_filter["filter_v"] = method(
                filter["self_attn_v_filter"][i], layer_filter_meta['filter_v']).to(device)
            layer_filter["bias_q"] = method(
                filter["self_attn_q_bias"][i], layer_filter_meta['bias_q']).to(device)
            layer_filter["bias_k"] = method(
                filter["self_attn_k_bias"][i], layer_filter_meta['bias_k']).to(device)
            layer_filter["bias_v"] = method(
                filter["self_attn_v_bias"][i], layer_filter_meta['bias_v']).to(device)
            layer_filter["filter_out"] = method(
                filter["self_attn_out_filter"][i], layer_filter_meta['filter_out']).to(device)
            layer_filter["bias_out"] = method(
                filter["self_attn_out_bias"][i], layer_filter_meta['bias_out']).to(device)
            layer_filter["ffn_filter1"] = method(
                filter["ffn_inner_filter"][i], layer_filter_meta['ffn_filter1']).to(device)
            layer_filter["ffn_bias1"] = method(
                filter["ffn_inner_bias"][i], layer_filter_meta['ffn_bias1']).to(device)
            layer_filter["ffn_filter2"] = method(
                filter["ffn_out_filter"][i], layer_filter_meta['ffn_filter2']).to(device)
            layer_filter["ffn_bias2"] = method(
                filter["ffn_out_bias"][i], layer_filter_meta['ffn_bias2']).to(device)

            if self.use_gated_ffn:
                if "ffn_gate_filter" not in filter:
                    raise Exception(
                        "use_gated_ffn is true, but ffn_gate_filter is not found in filter file")
                layer_filter["ffn_filter3"] = method(
                    filter["ffn_gate_filter"][i], layer_filter_meta['ffn_filter3']).to(device)
                layer_filter["ffn_bias3"] = method(
                    filter["ffn_gate_bias"][i], layer_filter_meta['ffn_bias3']).to(device)

            layer_filter["attn_layernorm_scale"] = method(
                filter["self_attn_layer_norm_scale"][i], layer_filter_meta['attn_layernorm_scale']).to(device)
            layer_filter["attn_layernorm_bias"] = method(
                filter["self_attn_layer_norm_bias"][i], layer_filter_meta['attn_layernorm_bias']).to(device)
            layer_filter["ffn_layernorm_scale"] = method(
                filter["ffn_layer_norm_scale"][i], layer_filter_meta['ffn_layernorm_scale']).to(device)
            layer_filter["ffn_layernorm_bias"] = method(
                filter["ffn_layer_norm_bias"][i], layer_filter_meta['ffn_layernorm_bias']).to(device)

            layer_filters.append(layer_filter)

        logit_layernorm_scale = method(
            filter["logit_layer_norm_scale"][0], input_bias_meta).to(device)
        logit_layernorm_bias = method(
            filter["logit_layer_norm_bias"][0], input_bias_meta).to(device)

        if self.is_embedding_weight_quantized:
            logit_filter_meta = (
                filter_data_type, (seg_vocab_size, input_size // filter_ci_div))
        else:
            logit_filter_meta = (data_type, (seg_vocab_size, input_size // 1))
        logit_filter = method(
            filter["logit_filter"][0], logit_filter_meta).to(device)

        if "logit_bias" in filter:
            logit_bias_meta = (data_type, (seg_vocab_size,))
            logit_bias = method(filter["logit_bias"]
                                [0], logit_bias_meta).to(device)
        else:
            logit_bias = None

        return layer_filters, logit_layernorm_scale, logit_layernorm_bias, logit_filter, logit_bias, position_embedding, zeros

    def load_single_gpt_tp_filters_from_file(self, map: dict, is_binary_file: bool):
        rank = self.rank

        def slice(tp_id, in_list):
            from_index = self.layer_num * tp_id
            to_index = self.layer_num * (tp_id + 1)
            return in_list[from_index:to_index]

        sub_map = {}
        for name, status in self.filter_status.items():
            if name not in map:
                continue

            expected_size = 1 * (self.tp_num if status['is_tp'] else 1) * (
                self.layer_num if status['is_per_layer'] else 1)
            assert len(
                map[name]) == expected_size, f"filter size error: {name} {len(map[name])} vs {expected_size}"

            if status['is_tp'] and status['is_per_layer']:
                sub_map[name] = slice(rank, map[name])
            elif status['is_tp'] and not status['is_per_layer']:
                sub_map[name] = [map[name][rank]]
            elif not status['is_tp'] and status['is_per_layer']:
                sub_map[name] = map[name]
            else:
                sub_map[name] = [map[name][0]]

        sub_filter = self.gpt_decoder_filter_from_file(sub_map, is_binary_file)
        return sub_filter
