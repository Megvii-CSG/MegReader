import torch

from concern.config import State
from .sequence_recognition_representer import SequenceRecognitionRepresenter


class CTCRepresenter2D(SequenceRecognitionRepresenter):
    max_size = State(default=32)

    def represent(self, batch, pred):
        '''
        This class is supposed to be used with
        the measurer `SequenceRecognitionMeasurer`
        '''
        classify, mask = pred
        labels = batch['label']

        '''
        classify: (N, C, H, W)
        mask: (N, 1, H, W)
        return:
            output: {
                'label_string': string of gt label,
                'pred_string': string of prediction
            }
        '''
        heatmap = classify * mask
        classify = classify.to('cpu')
        mask = mask.to('cpu')
        paths = heatmap.max(1, keepdim=True)[0].argmax(
            2, keepdim=True)  # (N, 1, 1, W)
        C = classify.size(1)
        paths = paths.repeat(1, C, 1, 1)  # (N, C, 1, W)
        selected_probabilities = heatmap.gather(2, paths)  # (N, C, W)
        pred = selected_probabilities.argmax(1).squeeze(1)  # (N, W)
        output = torch.zeros(
            pred.shape[0], pred.shape[-1], dtype=torch.int) + self.charset.blank
        pred = pred.to('cpu')
        output = output.to('cpu')

        for i in range(pred.shape[0]):
            valid = 0
            previous = self.charset.blank
            for j in range(pred.shape[1]):
                c = pred[i][j]
                if c == previous or c == self.charset.unknown:
                    continue
                if not c == self.charset.blank:
                    output[i][valid] = c
                    valid += 1
                previous = c

        result = []
        for i in range(labels.shape[0]):
            result.append({
                'label_string': self.label_to_string(labels[i]),
                'pred_string': self.label_to_string(output[i]),
                'mask': mask[i][0],
                'classify': classify[i]
            })

        return result
