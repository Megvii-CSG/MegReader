from concern.config import State
from .meta_loader import MetaLoader


class DetectionMetaLoader(MetaLoader):
    key = State(default='gt')
    scan_meta = True

    def __init__(self, key=None, cmd={}, **kwargs):
        super().__init__(cmd=cmd, **kwargs)
        if key is not None:
            self.key = key

    def parse_meta(self, data_id, meta):
        return dict(data_ids=data_id, gt=self.get_annotation(meta)[self.key])
