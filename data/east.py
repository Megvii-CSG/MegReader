import pickle

import cv2
import numpy as np
from shapely.geometry import Polygon

from concern.config import Configurable, State


class MakeEASTData(Configurable):
    shrink = State(default=0.5)
    background_weight = State(default=3.0)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def find_polygon_radius(self, poly):
        poly = Polygon(poly)
        low = 0
        high = 65536
        while high - low > 0.1:
            mid = (high + low) / 2
            area = poly.buffer(-mid).area
            if area > 0.1:
                low = mid
            else:
                high = mid
        return high

    def __call__(self, data, *args, **kwargs):
        image, label, meta = data
        lines = label['polys']
        image_id = meta['data_id']

        h, w = image.shape[:2]

        heatmap = np.zeros((h, w), np.float32)
        heatmap_weight = np.zeros((h, w), np.float32)
        densebox = np.zeros((8, h, w), np.float32)
        densebox_weight = np.zeros((h, w), np.float32)
        train_mask = np.ones((h, w), np.float32)

        densebox_anchor = np.indices((h, w))[::-1].astype(np.float32)

        for line in lines:
            poly = line['points']

            assert(len(poly) == 4)
            quad = poly

            radius = self.find_polygon_radius(poly)
            shrinked_poly = list(Polygon(poly).buffer(-radius * self.shrink).exterior.coords)[:-1]
            shrinked_poly_points = np.array([shrinked_poly], np.int32)

            cv2.fillConvexPoly(heatmap, shrinked_poly_points, 1.0)
            cv2.fillConvexPoly(densebox_weight, shrinked_poly_points, 1.0)

            for i in range(0, 4):
                for j in range(0, 2):
                    cv2.fillConvexPoly(densebox[i * 2 + j], shrinked_poly_points, float(quad[i][j]))

            if line['ignore']:
                cv2.fillConvexPoly(train_mask, np.array([poly], np.int32), 0.0)

        heatmap_neg = np.logical_and(heatmap == 0, train_mask)
        heatmap_pos = np.logical_and(heatmap > 0, train_mask)

        heatmap_weight[heatmap_neg] = self.background_weight
        heatmap_weight[heatmap_pos] = train_mask.sum() / max(heatmap_pos.sum(), train_mask.sum() * 0.05)

        densebox_weight = densebox_weight * train_mask

        densebox = densebox - np.tile(densebox_anchor, (4, 1, 1))

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        label = {
            'heatmap': heatmap[np.newaxis],
            'heatmap_weight': heatmap_weight[np.newaxis],
            'densebox': densebox,
            'densebox_weight': densebox_weight[np.newaxis],
        }
        meta = {
            'image_id': image_id,
            'lines': lines,
        }
        return image, label, pickle.dumps(meta)
