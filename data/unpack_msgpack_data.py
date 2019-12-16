import io

import cv2
import numpy as np
import nori2 as nori
import msgpack
import config
import os
import lmdb
from PIL import Image

from concern.config import Configurable, State


class UnpackMsgpackData(Configurable):
    mode = State(default='BGR')

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        if config.will_use_nori:
            self.fetcher = nori.Fetcher()
        elif config.will_use_lmdb:
            self.envs = []
            self.txns = {}
        if 'mode' in cmd:
            self.mode = cmd['mode']

    def convert_obj(self, obj):
        if isinstance(obj, dict):
            ndata = {}
            for key, value in obj.items():
                nkey = key.decode()
                if nkey == 'img':
                    img = Image.open(io.BytesIO(value))
                    img = np.array(img.convert('RGB'))
                    if self.mode == 'BGR':
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    nvalue = img
                else:
                    nvalue = self.convert_obj(value)
                ndata[nkey] = nvalue
            return ndata
        elif isinstance(obj, list):
            return [self.convert_obj(item) for item in obj]
        elif isinstance(obj, bytes):
            return obj.decode()
        else:
            return obj

    def convert(self, data):
        obj = msgpack.loads(data, max_str_len=2 ** 31)
        return self.convert_obj(obj)

    def __call__(self, data_id, meta=None):
        if meta is None:
            meta = {}
        item = self.convert(self.fetcher.get(data_id))
        item['data_id'] = data_id
        meta.update(item)
        return meta


class TransformMsgpackData(UnpackMsgpackData):
    meta_loader = State(default=None)

    def __init__(self, meta_loader=None, cmd={}, **kwargs):
        super().__init__(cmd=cmd, meta_loader=meta_loader, **kwargs)
        print('transform')
        self.meta_loader = cmd.get('meta_loader', self.meta_loader)

    def get_item(self, data_id, meta):
        if config.will_use_nori:
            item = self.fetcher.get(data_id)
        elif config.will_use_lmdb:
            db_path = meta['db_path']
            if db_path not in self.envs:
                path = os.path.join(db_path, '')
                env = lmdb.open(db_path, max_dbs=1, lock=False)
                db_image = env.open_db('image'.encode())
                self.envs.append(env)
                self.txns[db_path] = env.begin(db=db_image)
            item = self.txns[db_path].get(data_id)
        else:
            raise NotImplementedError
        return item

    def __call__(self, data_id, meta):
        item = self.get_item(data_id, meta)
        item = self.convert(item)
        image = item.pop('img').astype(np.float32)
        if self.meta_loader is not None:
            meta['extra'] = item
            data = self.meta_loader.parse_meta(data_id, meta)
        else:
            data = meta
            data.update(**item)
        data.update(image=image, data_id=data_id)
        return data
