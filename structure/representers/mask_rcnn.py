import pickle

from concern.config import Configurable


class MaskRCNNRepresenter(Configurable):
    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def represent(self, batch, pred):
        image, label, meta = batch

        output = {
            'meta': [pickle.loads(value) for value in meta],
            'polygons_pred': pred,
        }
        return output
