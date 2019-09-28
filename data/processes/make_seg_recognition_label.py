import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper
import torch
import torch.nn.functional as F
from concern.config import State
from concern.charsets import DefaultCharset
import ipdb
from .data_process import DataProcess


class MakeSegRecognitionLabel(DataProcess):
    charset = State(default=DefaultCharset())
    shrink = State(default=True)
    shrink_ratio = State(default=0.25)
    exempt_chars = State(default=list(''))
    shape = State(default=(16, 64))

    max_size = 256

    def process(self, data: dict):
        assert 'charboxes' in data, 'charboxes in data is required'
        ori_height, ori_width = data['image'].shape[:2]
        charboxes = data['charboxes']
        gt = data['gt']
        height, width = self.shape
        ratio_h, ratio_w = float(height) / ori_height, float(width) / ori_width
        assert len(charboxes) == len(gt)
        mask = np.zeros((height, width), dtype=np.float32)
        classify = np.zeros((height, width), dtype=np.int32)
        order_map = np.zeros((height, width), dtype=np.int32)
        shrink_ratio = self.shrink_ratio
        boxes = np.zeros_like(charboxes)
        boxes[..., 0] = (charboxes[..., 0] * ratio_w + 0.5).astype(np.int32)
        boxes[..., 1] = (charboxes[..., 1] * ratio_h + 0.5).astype(np.int32)
        for box_index, box in enumerate(boxes):
            class_code = self.charset.index(gt[box_index])
            if self.shrink:
                if self.charset.is_empty(class_code) or gt[box_index] in self.exempt_chars:
                    shrink_ratio = 0
                try:
                    rect = self.poly_to_rect(box, shrink_ratio)
                except AssertionError:
                    # invalid poly
                    continue
            else:
                rect = box
            if rect is not None:
                self.render_rect(mask, rect)
                self.render_rect(classify, rect, class_code)
                self.render_rect(order_map, rect, box_index + 1)
        data['mask'] = mask
        data['classify'] = classify
        if classify.sum() == 0:
            raise 'gt is empty!'
        data['ordermaps'] = order_map
        return data

    def poly_to_rect(self, poly, shrink_ratio=None):
        if shrink_ratio is None:
            shrink_ratio = self.shrink_ratio
        polygon_shape = Polygon(poly)
        distance = polygon_shape.area * \
                   (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in poly]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        shrinked = padding.Execute(-distance)
        if shrinked == []:
            return None
        return shrinked

    def render_rect(self, canvas, poly, value=1):
        poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
        return cv2.fillPoly(canvas, [poly], value)
        '''
        xmin, xmax = poly[:, 0].min(), poly[:, 0].max()
        ymin, ymax = poly[:, 1].min(), poly[:, 1].max()
        canvas[ymin:ymax+1, xmin:xmax+1] = value
        return canvas
        '''
