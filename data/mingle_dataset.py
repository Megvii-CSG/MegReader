import bisect

import torch.utils.data as data

from concern.config import Configurable, State


class MingleDataset(data.Dataset, Configurable):
    datasets = State(default=[])

    def __init__(self, cmd={}, **kwargs):
        self.load_all(cmd=cmd, **kwargs)

        ratios = []
        sizes = []
        indices = []
        self.data_sources = []
        for i in range(len(self.datasets)):
            dataset_dict = self.datasets[i]
            ratio = dataset_dict['ratio']
            size = len(dataset_dict['dataset'])
            if size == 0:
                continue
            indices.append(i)
            ratios.append(ratio)
            sizes.append(size)
            self.data_sources.append(dataset_dict['dataset'])

        ratio_sum = sum(ratios)
        ratios = [r / ratio_sum for r in ratios]
        total = sum(sizes)
        for index, ratio in enumerate(ratios):
            quota = ratio * total
            if sizes[index] < quota:
                total = int(sizes[index] / ratio + 0.5)

        milestones = []
        for ratio in ratios[:-1]:
            milestones.append(int(ratio * total + 0.5))
        self.milestones = milestones
        self.total = total
        print('total', self.total)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        dataset_index = bisect.bisect(self.milestones, index)
        dataset = self.data_sources[dataset_index]
        if dataset_index == 0:
            real_index = index
        else:
            real_index = index - self.milestones[dataset_index - 1]
        return dataset.__getitem__(real_index)
