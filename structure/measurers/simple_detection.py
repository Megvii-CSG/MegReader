import pickle

import editdistance
import numpy as np

from concern import AverageMeter
from concern.config import Configurable
from concern.icdar2015_eval.detection.iou import DetectionIoUEvaluator


class SimpleDetectionMeasurer(Configurable):
    def __init__(self, **kwargs):
        self.evaluator = DetectionIoUEvaluator()

    def validate_measure(self, batch, output):
        return self.measure(batch, output), [0]

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch[0].shape[0]).tolist()

    def measure(self, batch, output):
        batch_meta = output['meta']
        batch_pred_polys = output['polygons_pred']

        results = []
        for pred_polys, meta in zip(batch_pred_polys, batch_meta):
            gt = meta['lines']
            pred = [{'points': points} for points in pred_polys]
            result = self.evaluator.evaluate_image(gt, pred)
            results.append(result)

        return results

    def gather_measure(self, raw_metrics, logger=None):
        raw_metrics = [image_metrics for batch_metrics in raw_metrics for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        hmean = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        hmean.update(2 * result['precision'] * result['recall'] / (result['precision'] + result['recall']), n=len(raw_metrics))

        return {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
        }


class SimpleDetectionE2EMeasurer(Configurable):
    def __init__(self, **kwargs):
        self.evaluator = DetectionIoUEvaluator()

    def validate_measure(self, batch, output):
        return self.measure(batch, output), [0]

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch[0].shape[0]).tolist()

    def measure(self, batch, output):
        images_meta = batch[2]

        results = []
        for pred, meta in zip(output, images_meta):
            gt = pickle.loads(meta)['lines']
            pred = [{'points': line['poly'], **line} for line in pred['lines']]
            result = self.evaluator.evaluate_image(gt, pred)
            gt_to_pred = {pair['gt']: pair['det'] for pair in result['pairs']}
            distances = []
            for gt_id in range(len(gt)):
                if not gt[gt_id]['ignore']:
                    gt_text = gt[gt_id]['text']
                    if gt_id in gt_to_pred:
                        pred_text = pred[gt_to_pred[gt_id]]['text']
                    else:
                        pred_text = ''
                    distance = editdistance.eval(gt_text, pred_text)
                    distances.append(distance / len(gt_text))
            result['edit_distance'] = distances
            results.append(result)

        return results

    def gather_measure(self, raw_metrics, logger=None):
        raw_metrics = [image_metrics for batch_metrics in raw_metrics for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)
        precision = result['precision']
        recall = result['recall']
        hmean = 2 * precision * recall / (precision + recall)
        edit_distance = [dis for m in raw_metrics for dis in m['edit_distance']]
        edit_distance = sum(edit_distance) / len(edit_distance)

        mprecision = AverageMeter()
        mrecall = AverageMeter()
        mhmean = AverageMeter()
        medit_distance = AverageMeter()

        mprecision.update(precision, n=len(raw_metrics))
        mrecall.update(recall, n=len(raw_metrics))
        mhmean.update(hmean, n=len(raw_metrics))
        medit_distance.update(edit_distance, n=len(raw_metrics))

        return {
            'precision': mprecision,
            'recall': mrecall,
            'hmean': mhmean,
            'edit_distance': medit_distance,
        }
