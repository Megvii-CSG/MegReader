import functools
import cv2
import numpy as np
import pickle
import lmdb
import os
from .dataset import Dataset
from concern.config import Configurable, State
from concern.distributed import is_main


class LMDBDataset(Dataset, Configurable):
    r'''Dataset reading from lmdb.
    Args:
        lmdb_paths: Pattern or list, required, the path of `data.mdb`,
            e.g. `the/path/`, `['the/path/a/', 'the/path/b/']`
    '''
    lmdb_paths = State()

    def __init__(self, lmdb_paths=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.lmdb_paths = self.list_or_pattern(lmdb_paths) or self.lmdb_paths
        self.debug = cmd.get('debug', False)
        self.envs = []
        self.txns = {}
        self.prepare()
        self.truncated = False
        self.data = None

    def __del__(self):
        for env in self.envs:
            env.close()

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

        return self.prepare_meta_single(path_or_list)

    def prepare_meta_single(self, path_name):
        return self.meta_loader.load_meta(path_name)

    def prepare(self):
        self.meta = self.prepare_meta(self.lmdb_paths)
        if self.unpack is None:
            self.unpack = self.default_unpack

            # The fetcher is supposed to be initialized in the
            # sub-processes, or it will cause CRC Error.
            self.fetcher = None

        self.data_ids = self.meta.get('data_ids', self.meta.get('data_id', []))
        if self.debug:
            self.data_ids = self.data_ids[:32]
        self.num_samples = len(self.data_ids)

        # prepare lmdb environments
        for path in self.lmdb_paths:
            path = os.path.join(path, '')
            env = lmdb.open(path, max_dbs=1, lock=False)
            db_image = env.open_db('image'.encode())
            self.envs.append(env)
            self.txns[path] = env.begin(db=db_image)

        if is_main():
            print(self.num_samples, 'images found')
        return self

    def searchImage(self, data_id, path):
        maybeImage = self.txns[path].get(data_id)
        assert maybeImage is not None, 'image %s not found as %s' % (
            data_id, path)
        return maybeImage

    def default_unpack(self, data_id, meta):
        data = self.searchImage(data_id, meta['db_path'])
        image = np.fromstring(data, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR).astype('float32')
        meta['image'] = image
        return meta
