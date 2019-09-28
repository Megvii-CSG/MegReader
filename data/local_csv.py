import os

from torch.utils.data import Dataset as TorchDataset

import cv2
import numpy as np

from concern.config import Configurable, State


class LocalCSVDataset(TorchDataset, Configurable):
    csv_path = State()
    processes = State()
    debug = State(default=False)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(cmd=cmd, **kwargs)

        self.load_meta()
        self.debug = cmd.get('debug', False)

    def load_meta(self):
        self.meta = []
        for textline in open(self.csv_path):
            tokens = textline.strip().split('\t')
            filename = tokens[0]
            filepath = os.path.join(os.path.dirname(self.csv_path), filename)
            lines_coords = tokens[1::2]
            lines_text = tokens[2::2]
            lines = []
            for coords, text in zip(lines_coords, lines_text):
                poly = np.array(list(map(int, coords[1:-1].split(',')))).reshape(4, 2).tolist()
                lines.append({
                    'poly': poly,
                    'text': text,
                })
            self.meta.append({
                'img': cv2.imread(filepath, cv2.IMREAD_COLOR),
                'lines': lines,
                'filename': filename,
                'data_id': filename,
            })

        print(len(self.meta), 'images found')

    def __getitem__(self, index):
        data = self.meta[index]
        for process in self.processes:
            data = process(data)
        return data

    def __len__(self):
        return len(self.meta)
