import cv2
import numpy as np

from concern.config import Configurable, State


class EASTVisualizer(Configurable):
    vis_num = State(default=4)

    def __init__(self, **kwargs):
        pass

    def visualize_detection(self, image, heatmap, heatmask, densebox, polygons):
        image_show = image.transpose(1, 2, 0).copy()

        h, w = image_show.shape[:2]
        densebox_image = np.zeros((h, w, 3), 'uint8')
        densebox_anchor = np.indices((h, w))[::-1]

        colors = [(64, 64, 255), (64, 255, 255), (64, 255, 64), (255, 255, 64)]
        for i in range(0, 4):
            points = densebox[i * 2: i * 2 + 2] + densebox_anchor
            points = points[np.tile(heatmask[np.newaxis], (2, 1, 1)) > 0]
            points = points.reshape(2, -1).astype('int32')
            mask = np.logical_and.reduce([points[0] >= 0, points[0] < w, points[1] >= 0, points[1] < h])
            densebox_image[points[1, mask], points[0, mask]] = colors[i]

        image_show = cv2.polylines(image_show, np.array(polygons, 'int32'), True, (0, 0, 255), 1)

        result_image = np.concatenate([
            image_show,
            cv2.cvtColor((heatmap * 255).astype('uint8'), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor((heatmask * 255).astype('uint8'), cv2.COLOR_GRAY2BGR),
            densebox_image,
        ], axis=1)

        return result_image

    def visualize_weight(self, idx, output):
        heatmap_weight = output['heatmap_weight'][idx]
        densebox_weight = output['densebox_weight'][idx]

        result_image = np.concatenate([
            cv2.cvtColor(((heatmap_weight / heatmap_weight.max())[0] * 255).astype('uint8'), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor((densebox_weight[0] * 255).astype('uint8'), cv2.COLOR_GRAY2BGR),
        ], axis=1)

        return result_image

    def visualize_pred_detection(self, idx, output):
        image = output['image'][idx]
        heatmap = output['heatmap_pred'][idx]
        heatmask = output['heatmask_pred'][idx]
        densebox = output['densebox_pred'][idx]
        polygons = output['polygons_pred'][idx]

        return self.visualize_detection(image, heatmap, heatmask, densebox, polygons)

    def visualize_gt_detection(self, idx, output):
        image = output['image'][idx]
        heatmap = output['heatmap'][idx]
        heatmask = output['heatmask'][idx]
        densebox = output['densebox'][idx]
        polygons = output['polygons'][idx]

        return self.visualize_detection(image, heatmap, heatmask, densebox, polygons)

    def get_image(self, idx, output):
        return np.concatenate([
            self.visualize_gt_detection(idx, output),
            self.visualize_weight(idx, output),
            self.visualize_pred_detection(idx, output),
        ], axis=1)

    def gt_get_image(self, idx, output):
        return np.concatenate([
            self.visualize_gt_detection(idx, output),
            self.visualize_weight(idx, output),
        ], axis=1)

    def visualize(self, batch, output, interested):
        images = {}
        for i in range(min(self.vis_num, len(output['image']))):
            show = self.get_image(i, output)
            images['image_%d' % i] = show.astype(np.uint8)
        return images

    def visualize_batch(self, batch, output):
        images = {}
        for i in range(len(output['image'])):
            show = self.gt_get_image(i, output)
            images['image_%d' % i] = show.astype(np.uint8)
        return images
