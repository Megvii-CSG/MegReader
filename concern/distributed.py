import sys
import functools
import torch
import torch.distributed as dist
from concern.average_meter import AverageMeter
from collections import defaultdict
import pickle


def do(function_name, value):
    if world_size() < 2:
        return value

    function = getattr(sys.modules[__name__], function_name + '_single')
    if isinstance(value, (list, tuple)):
        return do_list(function, value)
    elif isinstance(value, dict):
        return do_dict(function, value)
    return function(value)


reduce = functools.partial(do, 'reduce')
#gather = functools.partial(do, 'gather')


def get_average_meter(function, item):
    meter_avg = torch.tensor(item.avg).reshape(1,).cuda()
    meter_count = torch.tensor(item.count).reshape(1,).cuda()
    meter_count = function(meter_count)
    meter_avg = function(meter_avg)
    metrics = AverageMeter()
    if is_main():
        for i in range(meter_avg.shape[0]):
            metrics.update(meter_avg[i].double().item(),
                           meter_count[i].int().item())
    return metrics


def do_list(function, value):
    new_list = []
    for item in value:
        if isinstance(item, list):
            item = do_list(function, item)
        elif isinstance(item, dict):
            item = do_dict(function, item)
        elif isinstance(item, torch.Tensor):
            item = item.cuda()
            item = function(item)
        elif isinstance(item, AverageMeter):
            item = get_average_meter(item)
        new_list.append(item)
    return new_list


def do_dict(function, value):
    new_dict = dict()
    for key, item in value.items():
        if isinstance(item, list):
            #item=[]
            print('key %s len %d value %s' % (key, len(item), item))
            item = do_list(function, item)
            print('key %s value finished %s' % (key, item))
        elif isinstance(item, dict):
            item = do_dict(function, item)
        elif isinstance(item, torch.Tensor):
            item = item.cuda()
            item = function(item)
        elif isinstance(item, AverageMeter):
            item = get_average_meter(function, item)
        new_dict[key] = item
    return new_dict


def gather_single(value):
    gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return torch.cat(gathered, dim=0)


def reduce_single(value):
    dist.reduce(value, dst=0, op=dist.ReduceOp.SUM)
    return value / world_size()


def is_main():
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def is_distributed():
    if not dist.is_available():
        return False
    return dist.is_initialized()


def world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def merge_dict_list(dict_list):
    """
    merge dict_list into one dict
    Args:
        dict_list: list of dict
    Returns:
        merged_dict:
    """
    assert isinstance(dict_list, list), 'dict_list must be list'
    assert len(dict_list) > 0, 'dict_list must have at least one element'
    merged_dict = defaultdict(list)
    for dic in dict_list:
        for key, value in dic.items():
            merged_dict[key].append(value)
    return merged_dict


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data):
    dict_list = all_gather(data)
    return merge_dict_list(dict_list)