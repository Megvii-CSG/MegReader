import pickle

import cv2
import numpy as np
from skimage.draw import polygon as drawpoly

from concern.textsnake import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin
from concern.config import Configurable, State


class MakeTextsnakeData(Configurable):
    n_disk = State(default=15)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def make_text_region(self, image, polygons):

        tr_mask = np.zeros(image.shape[:2], np.uint8)
        train_mask = np.ones(image.shape[:2], np.uint8)

        for polygon in polygons:
            cv2.fillPoly(tr_mask, [np.array(polygon['points'], np.int32)], color=(1,))
            # we will  ignore the text that can not be identified, so we need this train mask.
            if polygon['ignore']:
                cv2.fillPoly(train_mask, [np.array(polygon['points'], np.int32)], color=(0,))
        return tr_mask, train_mask

    def fill_polygon(self, mask, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0], mask.shape[1]))
        mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2, center_line, radius,
                              tcl_mask, radius_map, sin_map, cos_map, expand=0.2):

        count = 0
        begin = 0
        end = 1
        while count < radius[0] / 2 and begin < 5:
            count += norm2(center_line[begin + 1] - center_line[begin])
            begin += 1

        count = 0
        while count < radius[-1] / 2 and end < 6:
            count += norm2(center_line[-end] - center_line[-end - 1])
            end += 1

        for i in range(begin, len(center_line) - 1 - end):
            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = sideline1[i]
            top2 = sideline1[i + 1]
            bottom1 = sideline2[i]
            bottom2 = sideline2[i + 1]

            sin_theta = vector_sin(c2 - c1)
            cos_theta = vector_cos(c2 - c1)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            polygon = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_mask, polygon, value=1)
            self.fill_polygon(radius_map, polygon, value=radius[i])
            self.fill_polygon(sin_map, polygon, value=sin_theta)
            self.fill_polygon(cos_map, polygon, value=cos_theta)

    def disk_cover(self, points, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        bottoms = find_bottom(points)  # find two bottoms of this Text
        e1, e2 = find_long_edges(points, bottoms)  # find two long edge sequence

        inner_points1 = split_edge_seqence(points, e1, n_disk)
        inner_points2 = split_edge_seqence(points, e2, n_disk)
        # inverse one of long edge because original long edge is inversed
        inner_points2 = inner_points2[::-1]

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radius = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radius

    def __call__(self, data, *args, **kwargs):
        image, label, meta = data
        polygons = label['polys']
        image_id = meta['data_id']

        height, weight, _ = image.shape

        tcl_mask = np.zeros(image.shape[:2], np.uint8)
        radius_map = np.zeros(image.shape[:2], np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.ones(image.shape[:2], np.float32)

        Cnts = []
        cares = []
        for i, polygon in enumerate(polygons):
            Cnts.append(polygon['points'])
            if not polygon['ignore']:
                sideline1, sideline2, center_points, radius = self.disk_cover(polygon['points'], n_disk=self.n_disk)
                self.make_text_center_line(sideline1, sideline2, center_points, radius, tcl_mask, radius_map, sin_map,
                                           cos_map)
                cares.append(1)
            else:
                cares.append(0)
        tr_mask, train_mask = self.make_text_region(image, polygons)
        # if tr_mask.sum() < 9:
        #     tr_mask[0:3, 0:3] = 1

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        label = {
            'train_mask': train_mask,
            'tr_mask': tr_mask,
            'tcl_mask': tcl_mask,
            'radius_map': radius_map,
            'sin_map': sin_map,
            'cos_map': cos_map,
        }
        meta = {
            'image_id': image_id,
            'Height': height,
            'Width': weight,
            'Cnts': Cnts,
            'cares': cares
        }
        return image, label, pickle.dumps(meta)
