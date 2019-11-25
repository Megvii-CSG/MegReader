import lmdb
import pickle
from collections import defaultdict
from concern.config import Configurable, State


class LMDBMetaLoader(Configurable):
    cache = State()
    force_reload = State(default=False)

    scan_meta = True
    scan_data = False
    post_prosess = None



    def __init__(self, force_reload=None, cmd={}, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
        self.force_reload = cmd.get('force_reload', self.force_reload)
        
        # lmdb environments
        self.envs = {}
        self.txns  = {}
        if force_reload is not None:
            self.force_reload = force_reload
    
    def __del__(self):
        for path in self.envs:
            envs[path].close()

    def load_meta(self, lmdb_path):
        
        if not self.force_reload and self.cache is not None:
            print(lmdb_path)
            meta = self.cache.read(lmdb_path)
            if meta is not None:
                return meta
        meta_info = defaultdict(list)
        valid_count = 0

        if lmdb_path not in self.envs:
            env = lmdb.open(lmdb_path, max_dbs=1, lock=False)
            self.envs[lmdb_path] = env
            db_extra = env.open_db('extra'.encode())
            self.txns[lmdb_path] = env.begin(db=db_extra)

        txn = self.txns[lmdb_path]
        cursor = txn.cursor()
        for data_id, value in cursor:
            meta_instance = self.parse_meta(data_id, value)
            if meta_instance is None:
                continue
            meta_instance['db_path'] = lmdb_path
            valid_count += 1
            if valid_count % 100000 == 0:
                print("%d instances processd" % valid_count)
            for key in meta_instance:
                    meta_info[key].append(meta_instance[key])

        print(valid_count, 'instances found')
        if self.post_prosess is not None:
            meta_info = self.post_prosess(meta_info)

        if self.cache is not None:
            self.cache.save(lmdb_path, meta_info)

        return meta_info

    def parse_meta(self, data_id, meta):
        raise NotImplementedError

    def get_annotation(self, meta):
        return pickle.loads(meta)['extra']
