import torch

import config
from .sequence_recognition_representer import SequenceRecognitionRepresenter
from concern.config import State


class CTCRepresenter(SequenceRecognitionRepresenter):
    def represent(self, batch, pred):
        '''
        decode ctc using greedy search
        pred: (N, C, W)
        return:
            output: {
                'label_string': string of gt label,
                'pred_string': string of prediction
            }
        '''
        labels = batch['label']
        pred = torch.argmax(pred, dim=1)
        pred = pred.select(1, 0)  # N, W
        output = torch.zeros(
            pred.shape[0], pred.shape[-1], dtype=torch.int) + self.charset.blank
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
            label_str = self.label_to_string(labels[i])
            pred_str = self.label_to_string(output[i])
            result.append({
                'label_string': label_str,
                'pred_string': pred_str
            })

        return result
