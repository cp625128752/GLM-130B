import torch
import torch.distributed as dist

_model_para_group = None


def is_model_parallel_initailized():
    return _model_para_group is not None

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_device_count():
    return torch.mlu.device_count()


def get_device():
    return torch.mlu.current_device()


class ModelParallelGroup:

    def __init__(self):
        rank = get_rank()
        device = rank % get_device_count()
        torch.mlu.set_device(device)



def initialize(backend=dist.Backend.MPI):

    assert torch.mlu.is_available()
    assert not is_model_parallel_initailized(), \
        f'parallel group has been already initialized.'

    print('Initializing tensor and pipeline parallel...')
    dist.init_process_group(backend=backend)

    global _model_para_group
    _model_para_group = ModelParallelGroup()

