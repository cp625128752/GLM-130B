import os
from .config import GptConfig
from .constants import *
import torch.nn.functional as F
from typing import List
import torch
import torch_mlu
import torch.nn as nn
import numpy as np

def build_embedding(vocab_size:int, input_size:int, filter_value)->nn.Embedding:
    with torch.no_grad():
        embedding_layer = nn.Embedding(vocab_size, input_size)
        filter_value = torch.from_numpy(filter_value).to(torch.float16)
        embedding_layer.weight.data.copy_(filter_value)
        return embedding_layer

def get_mask(valid_seq_len:int, pad_seq_len:int)->np.ndarray:
    mask = []
    batch_size = len(valid_seq_len)
    mask.extend([-10000] * (batch_size * pad_seq_len * pad_seq_len))
    for i in range(batch_size):
        start_idx = i * pad_seq_len * pad_seq_len
        for j in range(valid_seq_len[i]):
            for k in range(valid_seq_len[i] - 1):
                mask[start_idx + j * pad_seq_len + k] = 0
                if j == valid_seq_len[i] - 1 and k == valid_seq_len[i] - 2:
                    mask[start_idx + j * pad_seq_len + k + 1] = 0
    return np.array(mask)


def get_chat_glm_pos_id(valid_seq_len: List[int], pad_seq_len: int, is_2d: bool) -> np.ndarray:
    batch_size = len(valid_seq_len)
    dimension = 2 if is_2d else 1
    pos_ids = np.zeros(batch_size * dimension *  pad_seq_len , dtype=np.int32)
    for i in range(batch_size):
        start_idx = i * (2 if is_2d else 1) * pad_seq_len
        for j in range(pad_seq_len):
            if j < valid_seq_len[i] - 1:
                pos_ids[start_idx + j] = j
                if is_2d:
                    pos_ids[start_idx + pad_seq_len + j] = 0
            else:
                pos_ids[start_idx + j] = valid_seq_len[i] - 2
                if is_2d:
                    pos_ids[start_idx + pad_seq_len + j] = 1
    pos_ids = pos_ids.reshape(batch_size,dimension,pad_seq_len)
    return pos_ids

def extract_file(cfg, suffix):
    files = []
    for entry in cfg.filter_files:
        if suffix in entry:
            files.append(entry)
    if len(files) > 1:
        raise RuntimeError(f"Found multiple files ending in {suffix}, unable to determine which one to read.")
    if not files:
        raise RuntimeError(f"Can't find file ending in {suffix}")
    return files[0]

def get_embedding_filters(gpt_config):
    print("==========Reading Embedding Filters==========")
    size = gpt_config.input_size * gpt_config.vocab_size
    if not gpt_config.is_mock_filter():
        filter_file = extract_file(gpt_config, ".embed_in.weight.bin")
        try:
            with open(filter_file, "rb") as file:
                filter_data = np.fromfile(file, dtype=np.float16, count=size)
            return filter_data
        except IOError:
            raise IOError("Can't open file: " + filter_file)
    elif not gpt_config.skip_embedding:
        return np.empty(size, dtype=np.float16)

# FIXME: 类型检查
def get_decoder_layer_input(input:torch.Tensor, gpt_config:GptConfig, is_context:bool, batch_seqlen:int=None, mask:int=None, token_offset:int=None):
    inputs = {}
    batch = gpt_config.once_batch_size

    assert input is not None and input.numel() > 0, "input is None or empty"

    if is_context:
        valid_token = None
        if batch_seqlen:
            valid_token = torch.tensor(batch_seqlen, dtype=torch.int32)
        if mask is not None and isinstance(mask,np.ndarray) and mask.size > 0:
            # FIXME: mask 是 fp16没错
            inputs["mask"] = torch.tensor(mask, dtype=torch.float16)
        inputs["valid_token"] = valid_token

    if not (token_offset is None or token_offset.size == 0):
        token_offset = torch.tensor(token_offset, dtype=torch.int32)
    else:
        token_offset_size = batch 
        if gpt_config.position_embedding_type in ["CNNL_ATTN_FOLD_ROTARY_EMBEDDING_2D" ,"CNNL_ATTN_CROSS_ROTARY_EMBEDDING_2D"]:
            token_offset_size *= 2
        # FIXME:  这里的空值后面是怎么处理的？
        token_offset = torch.zeros(token_offset_size, dtype=torch.int32)
    inputs["token_offset"] = token_offset

    inputs["input"] = input 
    return inputs

def invoke_slice_individual_posid(context_token_offset):
    assert isinstance(context_token_offset,torch.Tensor)
    assert len(context_token_offset.shape) == 3
    # batch = int(context_token_offset.shape[0])
    dimension = int(context_token_offset.shape[1])
    # seq_len = context_token_offset.shape[2]
    generate_token_offset = context_token_offset[:,:,-1:]
    print("context_token_offset shape: ",context_token_offset.shape)
    print("generate_token_offset shape: ",generate_token_offset.shape)
    if dimension > 1:
        generate_token_offset[:,1,-1] += 1
    return generate_token_offset

# FIXME:感觉还是肯定有问题
class GPTPostProcess(nn.Module):
    def __init__(self):
        super(GPTPostProcess, self).__init__()

    def forward(self, input_tensor,batch_seqlen,temperature, top_k, top_p,batch,is_context):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batch = batch
        self.is_context = is_context
        if self.is_context:
            concat_input_tensor = []
            for i in range(self.batch):
                valid_seq_len = batch_seqlen[i]
                assert input_tensor.shape[1] >= 1
                if input_tensor.shape[1] > 1:
                    last_token = input_tensor[i][valid_seq_len-1]
                else:
                    last_token = input_tensor[i][0]
                last_token = last_token.unsqueeze(0)
                concat_input_tensor.append(last_token)
            input_tensor = torch.cat(concat_input_tensor, dim=0)
        else:
            input_tensor = input_tensor.reshape((self.batch, 1, -1))
        if self.temperature == 0.0 and self.top_k == 0 and self.top_p == 0.0:
            output = torch.argmax(input_tensor, dim=-1,keepdim=True)
        else:
            input_tmp = input_tensor.float()
            if self.temperature > 0.0:
                div_value = torch.tensor([self.temperature], dtype=torch.float32)
                input_tmp = input_tmp / div_value
            filter_value = -torch.finfo(torch.float32).max
            if self.top_k > 0:
                k_value = [self.top_k]
                topk_out = torch.topk(input_tmp, k_value, dim=-1)
                # topk_out.values  这是什么写法？
                min_node = torch.min(topk_out.values, dim=-1, keepdim=True)[0]
                lt_node = input_tmp < min_node
                masked_fill_node = torch.where(lt_node, input_tmp, torch.tensor([filter_value], dtype=torch.float32))
                input_tmp = masked_fill_node
            if self.top_p > 0.0:
                sorted_values, sorted_indices = torch.sort(input_tmp, descending=True, dim=-1)
                softmax_node = F.softmax(sorted_values, dim=-1)
                cumsum_node = torch.cumsum(softmax_node, dim=-1)
                gt_value = [self.top_p]
                gt_node = cumsum_node > gt_value
                roll_node = torch.roll(gt_node, shifts=1, dims=-1)
                roll_node[0] = False
                index_fill_node = torch.index_fill(roll_node, dim=-1, index=0, value=False)
                concat_input = []
                for i in range(self.batch):
                    gather_node0 = torch.gather(index_fill_node, dim=-1, index=torch.tensor([i]))
                    gather_node1 = torch.gather(sorted_indices, dim=-1, index=torch.tensor([i]))
                    masked_select_node = torch.masked_select(gather_node1, gather_node0)
                    gather_node2 = torch.gather(input_tmp, dim=-1, index=torch.tensor([i]))
                    index_fill_node1 = torch.index_fill(gather_node2, dim=0, index=masked_select_node, value=filter_value)
                    unsqueeze_node1 = torch.unsqueeze(index_fill_node1, dim=-1)
                    concat_input.append(unsqueeze_node1)
                
                input_tmp = torch.cat(concat_input, dim=0)
            
            softmax_node1 = F.softmax(input_tmp, dim=-1)
            output = softmax_node1

        return output
