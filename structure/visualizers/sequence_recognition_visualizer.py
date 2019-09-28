import cv2
import numpy as np

from concern.config import Configurable, State
import concern.webcv2 as webcv2
from concern.charsets import DefaultCharset
from data.processes.normalize_image import NormalizeImage


class SequenceRecognitionVisualizer(Configurable):
    charset = State(default=DefaultCharset())

    def __init__(self, cmd={}, **kwargs):
        self.eager = cmd.get('eager_show', False)
        self.load_all(**kwargs)

    def visualize(self, batch, output, interested):
        return self.visualize_batch(batch, output)

    def visualize_batch(self, batch, output):
        images, labels, lengths = batch['image'], batch['label'], batch['length']
        for i in range(images.shape[0]):
            image = NormalizeImage.restore(images[i])
            gt = self.charset.label_to_string(labels[i])
            webcv2.imshow(output[i]['pred_string'] + '_' + str(i) + '_' + gt, image)
            # folder = 'images/dropout/lexicon/'
            # np.save(folder + output[i]['pred_string'] + '_' + gt + '_' + batch['data_ids'][i], image)
        webcv2.waitKey()
        return {
            'image': (np.clip(batch['image'][0].cpu().data.numpy().transpose(1, 2, 0) + 0.5, 0, 1) * 255).astype(
                'uint8')
        }
