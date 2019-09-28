import numpy as np

from concern.config import State
from concern.charsets import DefaultCharset

from .data_process import DataProcess


class MakeRecognitionLabel(DataProcess):
    charset = State(default=DefaultCharset())
    max_size = State(default=32)

    def process(self, data):
        assert 'gt' in data, '`gt` in data is required by this process'
        gt = data['gt']
        label = self.gt_to_label(gt)
        data['label'] = label
        if label.sum() == 0:
            raise 'Empty Label'  # FIXME: package into a class.

        length = len(gt)
        if self.max_size is not None:
            length = min(length, self.max_size)
        length = np.array(length, dtype=np.int32)
        data['length'] = length
        return data

    def gt_to_label(self, gt, image=None):
        if self.max_size is not None:
            return self.charset.string_to_label(gt)[:self.max_size]
        else:
            return self.charset.string_to_label(gt)
