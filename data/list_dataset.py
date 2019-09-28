import io
import os

from torch.utils.data import Dataset as TorchDataset

import cv2
import numpy as np
from matplotlib.pyplot import imread

from concern.config import Configurable, State


class ListDataset(TorchDataset, Configurable):
    list_file = State()
    processes = State()
    image_path = State()
    gt_path = State()

    def __init__(self,
                 debug=False,
                 list_file=None, image_path=None, gt_path=None,
                 **kwargs):
        self.load_all(**kwargs)

        self.image_path = image_path or self.image_path
        self.gt_path = gt_path or self.gt_path
        self.list_file = list_file or self.list_file
        self.gt_paths, self.image_paths =\
            self.load_meta()  # FIXME: this should be removed

    def load_meta(self):
        base_path = os.path.dirname(self.list_file)
        gt_base_path = os.path.join(base_path, 'gts')
        image_base_path = self.image_path
        gt_paths = []
        image_paths = []
        with open(self.list_file, 'rt') as list_reader:
            for _line in list_reader.readlines():
                line = _line.strip()
                gt_paths.append(os.path.join(gt_base_path, line + '.txt'))
                image_paths.append(os.path.join(image_base_path, line))
        print(len(gt_paths), 'images found')
        self.loaded = True
        return gt_paths, image_paths

    def __getitem__(self, index):
        if not self.loaded:
            self.load()
        gt_path = self.gt_paths[index]
        image_path = self.image_paths[index]
        data = (gt_path, image_path)
        for process in self.processes:
            data = process(data)
        return data

    def __len__(self):
        return len(self.gt_paths)


class UnpackTxtData(Configurable):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        gt_path, image_path = data
        image_name = os.path.basename(image_path)

        lines = []
        with open(gt_path) as reader:
            for line in reader.readlines():
                line = line.replace('\ufeff', '').strip()
                if '\xef\xbb\xbf' in line:
                    import pdb; pdb.set_trace()
                line_list = line.split(',')
                assert len(line_list) == 9
                points = np.array([float(scalar)
                                   for scalar in line_list[:-1]]).reshape(-1, 2)
                lines.append(dict(
                    poly=points.tolist(),
                    text=line_list[-1],
                    filename=image_name))

        return dict(
            img=imread(image_path, mode='RGB'),
            lines=lines,
            data_id=image_name,
            filename=image_name)
