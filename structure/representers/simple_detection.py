import pickle

import cv2
import numpy as np
import skimage
from shapely.geometry import Polygon, LineString

from concern.convert import to_np
from concern.config import Configurable, State
from data.simple_detection import binary_search_smallest_width


class SimpleDetectionRepresenter(Configurable):
    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def represent_batch(self, batch):
        image, label, meta = batch

        output = {
            'image': to_np(image),
            **{name: to_np(m) for name, m in label.items()},
            'meta': [pickle.loads(value) for value in meta],
        }
        self.postprocess(output)

        unrecovered_polygons = []
        polygons = []
        for idx in range(image.shape[0]):
            single_output = self.get_single_output(output, idx)
            p = self.get_polygons(single_output)
            unrecovered_polygons.append(p)
            polygons.append(self.recover_polygons(single_output, p))
        output['unrecovered_polygons'] = unrecovered_polygons
        output['polygons'] = polygons

        return output

    def get_single_output(self, output, idx):
        return {name: value[idx] for name, value in output.items()}

    def threshold_heatmap(self, heatmap, thr):
        return (heatmap > thr).astype('uint8')

    def postprocess(self, output):
        raise NotImplementedError()

    def get_polygons(self, output):
        raise NotImplementedError()

    def recover_polygons(self, output, polygons):
        scale_h = output['meta']['scale_h']
        scale_w = output['meta']['scale_w']

        recovered_polygons = []
        for polygon in polygons:
            recovered_polygons.append([(x / scale_w, y / scale_h) for x, y in polygon])
        return recovered_polygons

    def represent(self, batch, pred):
        image, label, meta = batch

        batch_output = self.represent_batch(batch)
        pred_output = {name: to_np(m) for name, m in pred.items()}
        self.postprocess(pred_output)

        unrecovered_polygons = []
        polygons = []
        for idx in range(image.shape[0]):
            batch_single_output = self.get_single_output(batch_output, idx)
            pred_single_output = self.get_single_output(pred_output, idx)
            p = self.get_polygons(pred_single_output)
            unrecovered_polygons.append(p)
            polygons.append(self.recover_polygons(batch_single_output, p))
        pred_output['unrecovered_polygons'] = unrecovered_polygons
        pred_output['polygons'] = polygons

        output = {
            **batch_output,
            **{name + '_pred': m for name, m in pred_output.items()},
        }
        return output


class SimpleSegRepresenter(SimpleDetectionRepresenter):
    heatmap_thr = State(default=0.5)
    min_average_score = State(default=0.8)
    min_area = State(default=200)
    max_poly_points = State(default=32)
    expand = State(default=2.0)
    max_polys = State(default=128)

    def postprocess(self, output):
        output['heatmask'] = self.threshold_heatmap(output['heatmap'], self.heatmap_thr)

    def get_polygons(self, output):
        heatmask = output['heatmask'][0]
        heatmap = output['heatmap'][0]

        _, contours, _ = cv2.findContours(heatmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        polygons = []
        for contour in contours[:self.max_polys]:
            poly = contour[:, 0]
            stride = len(poly) // self.max_poly_points + 1
            poly = poly[::stride]
            if len(poly) >= 3:
                height = binary_search_smallest_width(poly)
                expanded_poly = Polygon(poly).buffer(height * (self.expand - 1))
                if expanded_poly.geom_type == 'Polygon' and not expanded_poly.is_empty and expanded_poly.area > self.min_area:
                    poly_r, poly_c = skimage.draw.polygon(poly[:, 1], poly[:, 0])
                    if heatmap[poly_r, poly_c].mean() > self.min_average_score:
                        expanded_poly = list(expanded_poly.exterior.coords)[:-1]
                        polygons.append(expanded_poly)

        return polygons


class SimpleEASTRepresenter(SimpleDetectionRepresenter):
    heatmap_thr = State(default=0.5)

    def postprocess(self, output):
        output['heatmask'] = self.threshold_heatmap(output['heatmap'], self.heatmap_thr)

    def get_polygons(self, output):
        heatmask = output['heatmask'][0]
        densebox = output['densebox']

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


class SimpleTextsnakeRepresenter(SimpleDetectionRepresenter):
    heatmap_thr = State(default=0.5)
    min_area = State(default=200)

    def postprocess(self, output):
        output['heatmask'] = self.threshold_heatmap(output['heatmap'], self.heatmap_thr)

    def get_polygons(self, output):
        heatmask = output['heatmask'][0]
        radius = output['radius']

        _, contours, _ = cv2.findContours(heatmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        polygons = []
        for contour in contours:
            contour = contour[:, 0]
            mask = np.zeros(heatmask.shape[:2], dtype=np.uint8)
            for x, y in contour:
                r = radius[0, y, x]
                if r > 1:
                    cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)

            _, conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(conts) == 1:
                poly = conts[0][:, 0]
                if Polygon(poly).geom_type == 'Polygon' and Polygon(poly).area > self.min_area:
                    polygons.append(poly)

        return polygons


class SimpleMSRRepresenter(SimpleDetectionRepresenter):
    heatmap_thr = State(default=0.5)
    min_area = State(default=200)
    max_poly_points = State(default=32)

    def postprocess(self, output):
        output['heatmask'] = self.threshold_heatmap(output['heatmap'], self.heatmap_thr)

    def get_polygons(self, output):
        heatmask = output['heatmask'][0]
        offset = output['offset']
        _, contours, _ = cv2.findContours(heatmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        polygons = []
        for contour in contours:
            contour = contour[:, 0]
            poly = []
            stride = len(contour) // self.max_poly_points + 1
            for x, y in contour[::stride]:
                point = offset[:, y, x] + (x, y)
                poly.append(point)
            if len(poly) >= 3:
                if Polygon(poly).area > self.min_area:
                    polygons.append(poly)

        return polygons
