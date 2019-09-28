import numpy as np
import cv2

from concern.config import State

from .data_process import DataProcess


class MakeKeyPointMap(DataProcess):
    max_size = State(default=32)
    box_key = State(default='charboxes')
    shape = State(default=[16, 64])

    def process(self, data):
        assert self.box_key in data, '%s in data is required' % self.box_key

        ori_h, ori_w = data['image'].shape[:2]
        charboxes = data[self.box_key]
        height, width = self.shape
        ratio_h, ratio_w = float(height) / ori_h, float(width) / ori_w
        boxes = np.zeros_like(charboxes)
        boxes[..., 0] = (charboxes[..., 0] * ratio_w + 0.5).astype(np.int32)
        boxes[..., 1] = (charboxes[..., 1] * ratio_h + 0.5).astype(np.int32)
        charmaps = self.gen_keypoint_map((boxes).astype(np.float32), self.shape[0], self.shape[1])
        data['charmaps'] = charmaps
        return data

    def get_gaussian(self, h, w, m):
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        g_map = np.exp(-m * (np.power((xs * 2 / w - 1), 2) + np.power((ys * 2 / h - 1), 2)))
        return g_map

    def gen_keypoint_map(self, boxes, h, w):
        maps = np.zeros((self.max_size, h, w), dtype=np.float32)
        for ind, box in enumerate(boxes):
            _, _, box_w, box_h = cv2.boundingRect(box)
            src = np.array([[0, box_h], [box_w, box_h], [box_w, 0], [0, 0]]).astype(np.float32)
            g = self.get_gaussian(box_h, box_w, m=2)
            M = cv2.getPerspectiveTransform(src, box.astype(np.float32))
            maps[ind] = cv2.warpPerspective(g, M, (w, h))
        return maps
