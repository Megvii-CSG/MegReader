import sys
import functools
import torch
import torch.distributed as dist
from concern.average_meter import AverageMeter


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
            item = do_list(function, item)
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
