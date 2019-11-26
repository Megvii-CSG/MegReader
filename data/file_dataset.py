import functools

import cv2
from concern.config import Configurable, State
from concern.log import Log
from .dataset import Dataset
from concern.distributed import is_main


class FileDataset(Dataset, Configurable):
    r'''Dataset reading from files.
    Args:
        file_paths: Pattern or list, required, the file_paths containing data and annotations.
    '''
    file_paths = State()

    def __init__(self, path=None, file_paths=None, cmd={}, **kwargs):
        self.load_all(**kwargs)

        if file_paths is None:
            file_paths = path
        self.file_paths = self.list_or_pattern(file_paths) or self.file_paths

        self.debug = cmd.get('debug', False)
        self.prepare()

    def prepare(self):
        self.meta = self.prepare_meta(self.file_paths)

        if self.unpack is None:
            self.unpack = self.default_unpack

        self.data_ids = self.meta.get('data_ids', self.meta.get('data_id', []))
        if self.debug:
            self.data_ids = self.data_ids[:32]
        self.num_samples = len(self.data_ids)
        if is_main():
            print(self.num_samples, 'images found')
        return self

    def prepare_meta(self, path_or_list):
        def add(a_dict: dict, another_dict: dict):
            for key, value in another_dict.items():
                if key in a_dict:
                    a_dict[key] = a_dict[key] + value
                else:
                    a_dict[key] = value
            return a_dict

        if isinstance(path_or_list, list):
            return functools.reduce(add, [self.prepare_meta(path) for path in path_or_list])

        return self.prepare_meta_single(path_or_list)

    def default_unpack(self, data_id, meta):
        image = cv2.imread(data_id, cv2.IMREAD_COLOR).astype('float32')
        meta['image'] = image
        return meta
