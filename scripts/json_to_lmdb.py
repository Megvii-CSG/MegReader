import lmdb
import json
from fire import Fire
from collections import defaultdict
import os
import pickle
from tqdm import tqdm


def main(json_path=None, lmdb_path=None):
    assert json_path is not None, 'json_path is needed'
    if lmdb_path is None:
        lmdb_path = json_path

    meta = os.path.join(json_path, 'meta.json')
    data_ids = []
    value = {}
    env = lmdb.Environment(lmdb_path, subdir=True,
                           map_size=int(1e9), max_dbs=2, lock=False)
    db_extra = env.open_db('extra'.encode(), create=True)
    db_image = env.open_db('image'.encode(), create=True)
    with open(meta, 'r') as meta_reader:
        for line in tqdm(meta_reader):
            single_meta = json.loads(line)
            data_id = os.path.join(json_path, single_meta['filename'])
            data_id = str(data_id.encode('utf-8').decode('utf-8'))
            with open(data_id.encode(), 'rb') as file_reader:
                image = file_reader.read()
            value['extra'] = {}
            for key in single_meta['extra']:
                value['extra'][key] = single_meta['extra'][key]
            with env.begin(write=True) as lmdb_writer:
                lmdb_writer.put(data_id.encode(),
                                pickle.dumps(value), db=db_extra)
            with env.begin(write=True) as image_writer:
                image_writer.put(data_id.encode(), image, db=db_image)
    env.close()


if __name__ == "__main__":
    Fire(main)
