import numpy as np

from .data_process import DataProcess


class CharboxesFromTextlines(DataProcess):
    def process(self, data):
        text_lines = data['lines']

        charboxes = None
        shape = None
        for boxes in text_lines.charboxes:
            shape = boxes.shape[1:]
            if charboxes is None:
                charboxes = boxes
            else:
                charboxes = np.concatenate([charboxes, boxes], axis=0)
        charboxes = np.concatenate(charboxes, axis=0)
        if shape is not None:
            charboxes = charboxes.reshape(-1, *shape)
        data['charboxes'] = charboxes

        return data
