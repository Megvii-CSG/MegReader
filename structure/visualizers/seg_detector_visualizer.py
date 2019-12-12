import cv2
import concern.webcv2 as webcv2
import numpy as np
import torch

from concern.config import Configurable, State
from data.processes.make_icdar_data import MakeICDARData


class SegDetectorVisualizer(Configurable):
    vis_num = State(default=4)
    eager_show = State(default=False)

    def __init__(self, **kwargs):
        cmd = kwargs['cmd']
        if 'eager_show' in cmd:
            self.eager_show = cmd['eager_show']

    def visualize(self, batch, output_pair, interested):
        output, pred = output_pair
        result_dict = {}
        for i in range(batch['image'].size(0)):
            result_dict.update(
                self.single_visualize(batch, i, output[i], pred))
        if self.eager_show:
            webcv2.waitKey()
            return {}
        return result_dict

    def _visualize_heatmap(self, heatmap, canvas=None):
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        heatmap = (heatmap[0] * 255).astype(np.uint8)
        if canvas is None:
            pred_image = heatmap
        else:
            pred_image = (heatmap.reshape(
                *heatmap.shape[:2], 1).astype(np.float32) / 255 + 1) / 2 * canvas
            pred_image = pred_image.astype(np.uint8)
        return pred_image

    def single_visualize(self, batch, index, output, pred):
        image = batch['image'][index]
        polygons = batch['polygons'][index]
        if isinstance(polygons, torch.Tensor):
            polygons = polygons.cpu().data.numpy()
        ignore_tags = batch['ignore_tags'][index]
        original_shape = batch['shape'][index]
        filename = batch['filename'][index]
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        image = (image.cpu().numpy() * std + mean).transpose(1, 2, 0) * 255
        pred_canvas = image.copy().astype(np.uint8)
        original_shape = tuple(original_shape.tolist())
        pred_canvas = self._visualize_heatmap(pred['binary'][index], pred_canvas)

        if 'thresh' in pred:
            thresh = self._visualize_heatmap(pred['thresh'][index])

        if 'thresh_binary' in pred:
            thresh_binary = self._visualize_heatmap(pred['thresh_binary'][index])
            MakeICDARData.polylines(self, thresh_binary, polygons, ignore_tags)
        MakeICDARData.polylines(self, pred_canvas, polygons, ignore_tags)

        for box in output:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 1)
            if 'thresh_binary' in pred:
                cv2.polylines(thresh_binary, [box], True, (0, 255, 0), 1)

        if self.eager_show:
            webcv2.imshow(filename + ' output', cv2.resize(pred_canvas, (1024, 1024)))
            if 'thresh' in pred:
                webcv2.imshow(filename + ' thresh', cv2.resize(thresh, (1024, 1024)))
                webcv2.imshow(filename + ' pred', cv2.resize(pred_canvas, (1024, 1024)))
            if 'thresh_binary' in pred:
                webcv2.imshow(filename + ' thresh_binary', cv2.resize(thresh_binary, (1024, 1024)))
            return {}
        else:
            return {
                filename + '_output': pred_canvas,
                filename + '_pred': np.expand_dims(thresh_binary, 2) if thresh_binary is not None else None
            }
