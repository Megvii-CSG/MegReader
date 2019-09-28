import pickle

import cv2
import numpy as np

from concern.convert import to_np
from concern.config import Configurable, State


class EASTRepresenter(Configurable):
    heatmap_thr = State(default=0.5)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def get_polygons(self, heatmask, densebox):
        _, contours, _ = cv2.findContours(heatmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        polygons = []
        for contour in contours:
            points = []
            for x, y in contour[:, 0]:
                quad = densebox[:, y, x].reshape(4, 2) + (x, y)
                points.extend(quad)
            quad = cv2.boxPoints(cv2.minAreaRect(np.array(points, np.float32)))
            polygons.append(quad)

        return polygons

    def represent_batch(self, batch):
        image, label, meta = batch
        batch_size = image.shape[0]

        output = {
            'image': to_np(image),
            'heatmap': to_np(label['heatmap'][:, 0]),
            'heatmap_weight': to_np(label['heatmap_weight']),
            'densebox': to_np(label['densebox']),
            'densebox_weight': to_np(label['densebox_weight']),
            'meta': [pickle.loads(value) for value in meta],
        }
        output['heatmask'] = (output['heatmap'] > self.heatmap_thr).astype('uint8')

        output['polygons'] = [self.get_polygons(
            output['heatmask'][i],
            output['densebox'][i],
        ) for i in range(batch_size)]

        return output

    def represent(self, batch, pred):
        image, label, meta = batch
        batch_size = image.shape[0]

        output = self.represent_batch(batch)
        output = {
            **output,
            'heatmap_pred': to_np(pred['heatmap'][:, 0]),
            'densebox_pred': to_np(pred['densebox']),
        }
        output['heatmask_pred'] = (output['heatmap_pred'] > self.heatmap_thr).astype('uint8')
        output['polygons_pred'] = [self.get_polygons(
            output['heatmask_pred'][i],
            output['densebox_pred'][i],
        ) for i in range(batch_size)]

        return output
