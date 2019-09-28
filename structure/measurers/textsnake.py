import numpy as np

from concern import AverageMeter
from concern.config import Configurable
from concern.icdar2015_eval.detection.iou import DetectionIoUEvaluator


class TextsnakeMeasurer(Configurable):
    def __init__(self, **kwargs):
        self.evaluator = DetectionIoUEvaluator()

    def validate_measure(self, batch, output):
        return self.measure(batch, output), [0]

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch[0].shape[0]).tolist()

    def measure(self, batch, output):
        batch_meta = output['meta']
        batch_gt_polys = output['polygons_gt']
        batch_pred_polys = output['contours_pred']

        results = []
        for gt_polys, pred_polys, meta in zip(batch_gt_polys, batch_pred_polys, batch_meta):
            gt = [{'points': points, 'ignore': not cares} for points, cares in zip(gt_polys, meta['cares'])]
            pred = [{'points': points} for points in pred_polys]
            result = self.evaluator.evaluate_image(gt, pred)
            results.append(result)

        return results

    def gather_measure(self, raw_metrics, logger=None):
        raw_metrics = [image_metrics for batch_metrics in raw_metrics for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))

        return {
            'precision': precision,
            'recall': recall,
        }
