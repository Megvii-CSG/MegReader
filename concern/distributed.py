import sys
import functools
import torch
import torch.distributed as dist


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
gather = functools.partial(do, 'gather')


def do_list(function, value):
    new_list = []
    for item in value:
        item = item.cuda()
        item = function(item)
        new_list.append(item)
    return new_list


def do_dict(function, value):
    new_dict = dict()
    for key, item in value.items():
        item = item.cuda()
        item = function(item)
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
