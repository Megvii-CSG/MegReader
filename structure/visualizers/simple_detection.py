import cv2
import numpy as np

from concern.config import Configurable, State


class SimpleDetectionVisualizer(Configurable):
    vis_num = State(default=4)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def get_image_item(self):
        return 'image'

    def get_polygon_items(self):
        return ['unrecovered_polygons', 'unrecovered_polygons_pred']

    def get_offset_items(self):
        return []

    def get_heatmap_items(self):
        return []

    def get_weight_items(self):
        return []

    def draw_polygon_image(self, image, polygons):
        image = image.copy()
        for polygon in polygons:
            cv2.polylines(image, np.array([polygon], 'int32'), True, (0, 0, 255), 1)
        return image

    def draw_offset_image(self, image, offset, mask):
        image = image.copy()

        h, w = image.shape[:2]
        anchor = np.indices((h, w))[::-1]
        num_points = offset.shape[0] // 2

        color_table = np.zeros((num_points, 3), 'uint8')
        color_table[:, 0] = np.linspace(0, 255, num_points, dtype='uint8')
        color_table[:, 1] = 255
        color_table[:, 2] = 255
        color_table = cv2.cvtColor(color_table[np.newaxis], cv2.COLOR_HSV2BGR)[0]

        for i in range(0, num_points):
            points = offset[i * 2: i * 2 + 2] + anchor
            points = points[np.tile(mask, (2, 1, 1)) > 0]
            points = points.reshape(2, -1).astype('int32')
            inside_mask = np.logical_and.reduce([points[0] >= 0, points[0] < w, points[1] >= 0, points[1] < h])
            image[points[1, inside_mask], points[0, inside_mask]] = color_table[i]

        return image

    def draw_heatmap_image(self, heatmap):
        heatmap = heatmap[0]
        return cv2.cvtColor((heatmap * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

    def draw_weight_image(self, weight):
        weight = weight[0]
        return cv2.cvtColor(((weight / (weight.max() + 1e-8)) * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

    def draw_image(self, idx, output):
        image = output[self.get_image_item()][idx].transpose(1, 2, 0)
        images = []

        for item in self.get_polygon_items():
            if item in output:
                images.append(self.draw_polygon_image(image, output[item][idx]))

        for item, mask_item in self.get_offset_items():
            if item in output:
                images.append(self.draw_offset_image(image, output[item][idx], output[mask_item][idx]))

        for item in self.get_heatmap_items():
            if item in output:
                images.append(self.draw_heatmap_image(output[item][idx]))

        for item in self.get_weight_items():
            if item in output:
                images.append(self.draw_weight_image(output[item][idx]))

        return np.concatenate(images, axis=1)

    def visualize(self, batch, output, interested):
        images = {}
        for i in range(min(self.vis_num, len(output[self.get_image_item()]))):
            show = self.draw_image(i, output)
            images['image_%d' % i] = show.astype(np.uint8)
        return images

    def visualize_batch(self, batch, output):
        return self.visualize(batch, output, None)


class SimpleSegVisualizer(SimpleDetectionVisualizer):
    def get_heatmap_items(self):
        return ['heatmap', 'heatmap_pred']

    def get_weight_items(self):
        return ['heatmap_weight']


class SimpleEASTVisualizer(SimpleDetectionVisualizer):
    def get_offset_items(self):
        return [
            ('densebox', 'heatmask'),
            ('densebox_pred', 'heatmask_pred'),
        ]

    def get_heatmap_items(self):
        return ['heatmap', 'heatmap_pred']

    def get_weight_items(self):
        return ['heatmap_weight', 'densebox_weight']


class SimpleTextsnakeVisualizer(SimpleDetectionVisualizer):
    def get_heatmap_items(self):
        return ['heatmap', 'heatmap_pred']

    def get_weight_items(self):
        return ['heatmap_weight', 'radius_weight', 'radius', 'radius_pred']


class SimpleMSRVisualizer(SimpleDetectionVisualizer):
    def get_offset_items(self):
        return [
            ('offset', 'heatmask'),
            ('offset_pred', 'heatmask_pred'),
        ]

    def get_heatmap_items(self):
        return ['heatmap', 'heatmap_pred']

    def get_weight_items(self):
        return ['heatmap_weight', 'offset_weight']
