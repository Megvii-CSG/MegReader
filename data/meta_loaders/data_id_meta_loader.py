from concern.config import State
from .meta_loader import MetaLoader


class DataIdMetaLoader(MetaLoader):
    return_dict = State(default=False)
    scan_meta = False

    def __init__(self, return_dict=None, cmd={}, **kwargs):
        super().__init__(cmd=cmd, **kwargs)
        if return_dict is not None:
            self.return_dict = return_dict

    def parse_meta(self, data_id):
        return dict(data_id=data_id)

    def post_prosess(self, meta):
        if self.return_dict:
            return meta
        return meta['data_id']
