from hanziconv import HanziConv

from concern.charset_tool import stringQ2B
from concern.config import State
from .meta_loader import MetaLoader


class RecognitionMetaLoader(MetaLoader):
    skip_vertical = State(default=False)
    case_sensitive = State(default=False)
    simplify = State(default=False)
    key = State(default='words')
    scan_meta = True

    def __init__(self, key=None, cmd={}, **kwargs):
        super().__init__(cmd=cmd, **kwargs)
        if key is not None:
            self.key = key

    def may_simplify(self, words):
        garbled = stringQ2B(words)
        if self.simplify:
            return HanziConv.toSimplified(garbled)
        return garbled

    def parse_meta(self, data_id, meta):
        word = self.may_simplify(self.get_annotation(meta)[self.key])
        vertical = self.get_annotation(meta).get('vertical', False)
        if self.skip_vertical and vertical:
            return None

        if word == '###':
            return None
        return dict(data_ids=data_id, gt=word)
