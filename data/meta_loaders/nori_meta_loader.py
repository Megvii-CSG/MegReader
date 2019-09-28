from concern.config import Configurable, State
import nori2 as nori


class NoriMetaLoader(Configurable):
    cache = State()
    force_reload = State(default=False)

    scan_meta = True
    scan_data = False
    post_prosess = None

    def __init__(self, force_reload=None, cmd={}, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
        self.force_reload = cmd.get('force_reload', self.force_reload)
        if force_reload is not None:
            self.force_reload = force_reload

    def load_meta(self, nori_path):
        if not self.force_reload and self.cache is not None:
            meta = self.cache.read(nori_path)
            if meta is not None:
                return meta

        meta_info = dict()
        valid_count = 0
        with nori.open(nori_path) as reader:
            for data_id, data, meta in reader.scan(
                    scan_data=self.scan_data, scan_meta=self.scan_meta):
                args_tuple = (data_id, )
                if self.scan_data:
                    args_tuple = tuple((*args_tuple, data))
                if self.scan_meta:
                    args_tuple = tuple((*args_tuple, meta))
                meta_instance = self.parse_meta(*args_tuple)
                if meta_instance is None:
                    continue
                valid_count += 1
                if valid_count % 100000 == 0:
                    print("%d instances processd" % valid_count)
                for key in meta_instance:
                    the_list = meta_info.get(key, [])
                    the_list.append(meta_instance[key])
                    meta_info[key] = the_list

        print(valid_count, 'instances found')
        if self.post_prosess is not None:
            meta_info = self.post_prosess(meta_info)

        if self.cache is not None:
            self.cache.save(nori_path, meta_info)

        return meta_info

    def parse_meta(self, data_id, meta):
        raise NotImplementedError

    def get_annotation(self, meta):
        return meta['extra']
