from collections import OrderedDict

import torch

import config
from concern.config import State
from .ctc_representer import CTCRepresenter
from concern.charsets import Charset
import concern.webcv2 as webcv2
from data.nori_dataset import NoriDataset


class EnsembleCTCRepresenter(CTCRepresenter):
    '''decode multiple ensembled ctc models.
    '''
    charsets = State(default={})
    offset = State(default=0.5)

    def get_charset(self, key):
        return self.charsets.get(key, self.charset)

    def one_hot_to_chars(self, score: torch.Tensor, charset: Charset):
        '''
        Args:
            score: (C, 1, W)
            charset: The corresponding charset.
        Return:
            chars: the chars with maximum scores.
            scores: corresponding scores.
        '''
        pred = torch.argmax(score, dim=0)
        pred = pred[0]
        chars = []
        scores = []

        for w in range(pred.shape[0]):
            chars.append(charset[pred[w]])
            scores.append(score[pred[w], 0, w])
        return chars, scores

    def represent(self, batch, preds: dict, max_size=config.max_size):
        _, labels, _ = batch
        batch_size = labels.shape[0]

        output = []
        string_scores = []
        for batch_index in range(batch_size):
            pred_sequences = OrderedDict()
            pred_scores = OrderedDict()
            for key, pred in preds.items():
                chars, scores = self.one_hot_to_chars(
                    pred[batch_index], self.get_charset(key))
                pred_sequences[key] = chars
                pred_scores[key] = scores

            result_string, pred_score = self.merge_encode(pred_sequences, pred_scores)
            output.append(result_string)
            string_scores.append(pred_score)

        result = []
        for i in range(labels.shape[0]):
            result.append({
                'label_string': self.label_to_string(labels[i]),
                'pred_string': ''.join(output[i]),
                'score': string_scores[i],
            })

        return result

    def merge_encode(self, sequences, scores):
        result = []
        previous = self.charset.blank_char
        main_sequence = sequences['main']

        score_sum = 0
        for index, _char in enumerate(main_sequence):
            char, score = self.choose_char(_char,
                                    [s[index] for s in sequences.values()],
                                    [s[index] for s in scores.values()])

            if char == previous or self.charset.is_empty_char(char):
                previous = char
                continue
            else:
                previous = char
                result.append(previous)
                score_sum += score
        if score_sum > 0:
            score_sum /= len(result)
        return result, score_sum

    def choose_char(self, char, substitudes, scores):
        # if not self.charset.is_empty_char(char):
        #     return char

        max_score = -1
        index = None
        for i, (char, score) in enumerate(zip(substitudes, scores)):
            if self.charset.is_empty_char(char):
                score -= self.offset
            if score > max_score:
                max_score = score
                index = i
        return substitudes[index], max_score
