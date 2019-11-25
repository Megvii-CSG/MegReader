import pickle
import lmdb
import nori2 as nori
from tqdm import tqdm
from fire import Fire



def main(nori_path, lmdb_path=None):
    if lmdb_path is None:
        lmdb_path = nori_path
    env = lmdb.Environment(lmdb_path, map_size=int(5e10), writemap=True, max_dbs=2, lock=False)
    fetcher = nori.Fetcher(nori_path)
    db_extra = env.open_db('extra'.encode(), create=True)
    db_image = env.open_db('image'.encode(), create=True)
    with nori.open(nori_path, 'r') as nr:
        with env.begin(write=True) as writer:
            for data_id, data, meta in tqdm(nr.scan()):
                value = {}
                image = fetcher.get(data_id)
                value['extra'] = {}
                for key in meta['extra']:
                    value['extra'][key] = meta['extra'][key]
                writer.put(data_id.encode(), pickle.dumps(value), db=db_extra)
                writer.put(data_id.encode(), image, db=db_image)
    env.close()
    print('Finished')

if __name__ == '__main__':
    Fire(main)