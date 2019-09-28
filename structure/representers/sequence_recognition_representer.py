import torch

import config
from concern.config import Configurable, State
from concern.charsets import DefaultCharset

import cv2
import numpy as np
import concern.webcv2 as webcv


class SequenceRecognitionRepresenter(Configurable):
    charset = State(default=DefaultCharset())

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)

    def label_to_string(self, label):
        return self.charset.label_to_string(label)

    def represent(self, batch, pred):
        images, labels = batch['image'], batch['label']
        mask = torch.ones(pred.shape[0], dtype=torch.int).to(pred.device)

        for i in range(pred.shape[1]):
            mask = (
                1 - (pred[:, i] == self.charset.blank).type(torch.int)) * mask
            pred[:, i] = pred[:, i] * mask + self.charset.blank * (1 - mask)

        output = []
        for i in range(labels.shape[0]):
            label_str = self.label_to_string(labels[i])
            pred_str = self.label_to_string(pred[i])
            if False and label_str != pred_str:
                print('label: %s , pred: %s' % (label_str, pred_str))
                img = (np.clip(images[i].cpu().data.numpy().transpose(
                    1, 2, 0) + 0.5, 0, 1) * 255).astype('uint8')
                webcv.imshow('【 pred: <%s> , label: <%s> 】' % (
                    pred_str, label_str), np.array(img, dtype=np.uint8))
                if webcv.waitKey() == ord('q'):
                    continue
            output.append({
                'label_string': label_str,
                'pred_string': pred_str
            })

        return output


class SequenceRecognitionEvaluationRepresenter(Configurable):
    charset = State(default=DefaultCharset())

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)

    def label_to_string(self, label):
        return self.charset.label_to_string(label)

    def represent(self, batch, pred):
        images, labels, lengths = batch
        mask = torch.ones(pred.shape[0], dtype=torch.int)

        for i in range(pred.shape[1]):
            mask = (
                1 - (pred[:, i] == self.charset.blank).type(torch.int)) * mask
            pred[:, i] = pred[:, i] * mask + self.charset.blank * (1 - mask)

        output = []
        for i in range(images.shape[0]):
            pred_str = self.label_to_string(pred[i])
            output.append(pred_str)
        return output
