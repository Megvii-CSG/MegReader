import numpy as np

from concern.config import State
from .recognition_meta_loader import RecognitionMetaLoader


class CharboxMetaLoader(RecognitionMetaLoader):
    charbox_key = State(default='charboxes')
    transpose = State(default=False)

    def __init__(self, charbox_key=None, transpose=None, cmd={}, **kwargs):
        super().__init__(cmd=cmd, **kwargs)
        print('load CharBox')
        if charbox_key is not None:
            self.charbox_key = charbox_key
        if transpose is not None:
            self.transpose = transpose

    def parse_meta(self, data_id, meta):
        parsed = super().parse_meta(data_id, meta)
        if parsed is None:
            return

        charbox = np.array(self.get_annotation(meta)[self.charbox_key])
        if self.transpose:
            charbox = charbox.transpose(2, 1, 0)
        parsed['charboxes'] = charbox
        return parsed
