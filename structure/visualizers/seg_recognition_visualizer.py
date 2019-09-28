import cv2
import numpy as np

import concern.webcv2 as webcv2
from concern.config import Configurable, State
from concern.visualizer import Visualize
from data.processes import NormalizeImage


class SegRecognitionVisualizer(Configurable):
    eager_show = State(default=False)

    def __init__(self, cmd, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
        self.eager_show = cmd.get('eager_show', self.eager_show)

    def visualize(self, batch, output, interested):
        return self.visualize_batch(batch, output)

    def visualize_batch(self, batch, output):
        visualization = dict()
        for index, output_dict in enumerate(output):
            image = batch['image'][index]
            image = NormalizeImage.restore(image)

            mask = output_dict['mask']
            mask = cv2.resize(Visualize.visualize_weights(mask), image.shape[:2][::-1])

            classify = output_dict['classify']
            classify = cv2.resize(Visualize.visualize_heatmap(classify, format='CHW'), image.shape[:2][::-1])

            canvas = np.concatenate([image, mask, classify], axis=0)
            key = "【%s-%s】" % (output_dict['label_string'], output_dict['pred_string'])
            vis_dict = {
                key: canvas
            }

            if self.eager_show:
                for k, v in vis_dict.items():
                    # if output_dict['label_string'] != output_dict['pred_string']:
                    webcv2.imshow(k, v)
            visualization.update(vis_dict)
        if self.eager_show:
            webcv2.waitKey()
        return visualization
