import os
import json
from concern.config import Configurable, State


class JsonMetaLoader(Configurable):
    cache = State()
    force_reload = State(default=False)
    image_folder = State(default='images')
    json_file = State(default='meta.json')

    scan_meta = True
    scan_data = False
    post_prosess = None

    def __init__(self, force_reload=None, cmd={}, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
        self.force_reload = cmd.get('force_reload', self.force_reload)
        if force_reload is not None:
            self.force_reload = force_reload

    def load_meta(self, json_path):
        if not self.force_reload and self.cache is not None:
            meta = self.cache.read(json_path)
            if meta is not None:
                return meta

        meta_info = dict()
        valid_count = 0
        with open(os.path.join(json_path, self.json_file)) as reader:
            for line in reader.readlines():
                line = line.strip()
                single_meta = json.loads(line)
                data_id = os.path.join(json_path, single_meta['filename'])
                args_dict = dict(data_id=data_id)

                if self.scan_data:
                    with open(self.same_dir_with(json_path), self.image_folder) as reader:
                        data = reader.read()
                    args_dict.update(data=data)
                elif self.scan_meta:
                    args_dict.update(meta=single_meta)

                meta_instance = self.parse_meta(**args_dict)
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
            self.cache.save(json_path, meta_info)

        return meta_info

    def parse_meta(self, data_id, meta):
        raise NotImplementedError

    def get_annotation(self, meta):
        return meta['extra']

    def same_dir_with(self, full_path, dest):
        return os.path.join(os.path.dirname(full_path), dest)
