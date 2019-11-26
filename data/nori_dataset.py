import functools
import cv2
import numpy as np

from .dataset import Dataset
from concern.config import Configurable, State
from concern.nori_reader import NoriReader
from concern.distributed import is_main


class NoriDataset(Dataset, Configurable):
    r'''Dataset reading from nori.
    Args:
        nori_paths: Pattern or list, required, the noris containing data,
            e.g. `the/path/*.nori`, `['a.nori', 'b.nori']`
    '''
    nori_paths = State()

    def __init__(self, nori_paths=None, cmd={}, **kwargs):
        self.load_all(**kwargs)

        self.nori_paths = self.list_or_pattern(nori_paths) or self.nori_paths
        self.debug = cmd.get('debug', False)

        self.prepare()
        self.truncated = False
        self.data = None

    def prepare_meta(self, path_or_list):
        def add(a_dict: dict, another_dict: dict):
            new_dict = dict()
            for key, value in another_dict.items():
                if key in a_dict:
                    new_dict[key] = a_dict[key] + value
                else:
                    new_dict[key] = value
            return new_dict

        if isinstance(path_or_list, list):
            return functools.reduce(add, [self.prepare_meta(path) for path in path_or_list])

        assert type(path_or_list) == str, path_or_list
        assert path_or_list.endswith('.nori') or path_or_list.endswith('.nori/')
        return self.prepare_meta_single(path_or_list)

    def prepare_meta_single(self, path_name):
        return self.meta_loader.load_meta(path_name)

    def prepare(self):
        self.meta = self.prepare_meta(self.nori_paths)
        if self.unpack is None:
            self.unpack = self.default_unpack

            # The fetcher is supposed to be initialized in the
            # sub-processes, or it will cause CRC Error.
            self.fetcher = None

        self.data_ids = self.meta.get('data_ids', self.meta.get('data_id', []))
        if self.debug:
            self.data_ids = self.data_ids[:32]
        self.num_samples = len(self.data_ids)
        if is_main():
            print(self.num_samples, 'images found')
        return self

    def default_unpack(self, data_id, meta):
        if self.fetcher is None:
            self.fetcher = NoriReader(self.nori_paths)
        data = self.fetcher.get(data_id)
        image = np.fromstring(data, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR).astype('float32')
        meta['image'] = image
        return meta
