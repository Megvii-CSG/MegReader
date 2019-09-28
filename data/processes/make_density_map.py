import numpy as np
import cv2
from shapely.geometry import box
from scipy.optimize import broyden1 as solver

from concern.config import State
from .data_process import DataProcess
import concern.webcv2 as webcv2
from concern.visualizer import Visualize


class MakeDensityMap(DataProcess):
    '''
    Make density map from TextLines.
    '''
    size = State(default=None)

    def process(self, data):
        text_lines = data['lines']
        size = self.size
        if size is None:
            size = data['image'].shape[:2]

        density_map = np.zeros((*size, 2), dtype=np.float32)
        density_map[:, :, 0] = 1 / size[0]
        density_map[:, :, 1] = 1 / size[1]

        for text, rect in zip(text_lines.texts, text_lines.rects):
            '''
            rect: [xmin, ymin, xmax, ymax] * n
            '''
            cv2.rectangle(data['image'], (int(rect[0]), int(rect[1])),
                          (int(rect[2]), int(rect[3])), (0, 0, 255))
            self.render_constant(density_map[:, :, 0], *rect.tolist(),
                                 len(text) / (rect[2] - rect[0]))
            self.render_constant(density_map[:, :, 1], *rect.tolist(),
                                 1 / (rect[3] - rect[1]))
        data['density_map'] = density_map
        return data

    def render_constant(self, canvas, xmin, ymin, xmax, ymax, value=1):
        canvas[int(ymin+0.5):int(ymax+0.5)+1, int(xmin+0.5):int(xmax+0.5)+1] += value
        return canvas


class OptimizeArrangement(DataProcess):

    def process(self, data):
        text_lines = data['lines']

        rects = text_lines.rects
        centers = np.stack(((rects[:, 0] + rects[:, 2]) / 2,
                           (rects[:, 1] + rects[:, 3]) / 2), axis=1)  # Nx2
        min_height = 1
        ratios = []
        for rect in rects:
            width = (rect[2] - rect[0])
            height = (rect[3] - rect[1])
            ratio = width / height
            ratios.append(ratio)
            if ratio < 1:
                min_height = 1 / ratio

        ratios = np.array(ratios)
        heights = np.empty([rects.shape[0]])
        heights.fill(min_height)
        widths = ratios * heights
        shapes = np.stack((widths, heights), axis=1) / 2
        min_rects = np.zeros_like(rects)
        min_rects[:, :2] = centers - shapes
        min_rects[:, 2:] = centers + shapes

        for rect in rects:
            data['image'] = Visualize.visualize_rect(data['image'], rect, color=(0, 255, 0))

        for rect in min_rects:
            data['image'] = Visualize.visualize_rect(data['image'], rect)


        best_ratio = 100
        area = data['image'].shape[0] * data['image'].shape[1]

        def F(r):
            new_rects = self.expand(min_rects, r)
            return self.loss(new_rects, area)


        best_ratio = solver(F, best_ratio, f_tol=1e-2, maxiter=1000)
        print(best_ratio)
        best_rects = self.expand(min_rects, best_ratio)
        for rect in best_rects:
            data['image'] = Visualize.visualize_rect(data['image'], rect, color=(255, 0, 0))

        data['min_rects'] = min_rects
        data['rects'] = rects
        return data

    def iou(self, rect1, rect2):
        rect1 = box(*rect1)
        rect2 = box(*rect2)

        intersection = rect1.intersection(rect2).area
        union = rect1.union(rect2).area

        return intersection / union

    def union_area(self, rects):
        union_rect = None
        for rect in rects:
            if union_rect is None:
                union_rect = box(*rect)
            else:
                union_rect = union_rect.union(box(*rect))
        if union_rect is None:
            return 0
        return union_rect.area

    def loss(self, rects, area):
        iou = 0
        for i in range(len(rects)):
            for j in range(len(rects)):
                if i == j:
                    continue
                iou += self.iou(rects[i], rects[j])

        return iou * 10 + 1 - self.union_area(rects) / area

    def expand(self, rects, ratio):
        widths = (rects[:, 2] - rects[:, 0]) * ratio
        heights = (rects[:, 3] - rects[:, 1]) * ratio
        centers = np.stack(((rects[:, 0] + rects[:, 2]) / 2,
                           (rects[:, 1] + rects[:, 3]) / 2), axis=1)  # Nx2
        shapes = np.stack((widths, heights), axis=1) / 2
        new_rects = np.zeros_like(rects)
        new_rects[:, :2] = centers - shapes
        new_rects[:, 2:] = centers + shapes
        return new_rects
