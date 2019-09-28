import numpy as np

import config
from concern import AverageMeter, Logger
from concern.config import Configurable, State
import concern.webcv2 as webcv2
import editdistance as ed
import warnings


class SequenceRecognitionMeasurer(Configurable):
    nori_lexicon_path = State(default=None)

    def __init__(self, *args, **kwargs):
        self.load_all(*args, **kwargs)
        if self.nori_lexicon_path:
            with open(self.nori_lexicon_path) as f:
                self.nori_lexicon = set(f.read().split())
        else:
            self.nori_lexicon = None

    def validate_measure(self, batch, output):
        interested = []
        return self.measure(batch, output), interested

    evaluate_measure = validate_measure

    def measure(self, batch, output):
        edit_distance = self.edit_distance(batch, output)
        accuracy = self.accuracy(batch, output)
        if self.nori_lexicon:
            in_lexicon = self.in_lexicon(batch, output)
            return dict(accuracy=accuracy, edit_distance=edit_distance, in_lexicon=in_lexicon)
        else:
            return dict(accuracy=accuracy, edit_distance=edit_distance)

    def gather_measure(self, raw_metrics, logger: Logger = None):
        if not self.nori_lexicon:
            edit_distance = self.gather_edit_distance([(m['edit_distance']) for m in raw_metrics])
            accuracy = self.gather_accuracy([(m['accuracy']) for m in raw_metrics])
            return dict(accuracy=accuracy, edit_distance=edit_distance)
        else:
            total_edit_distance, in_lexicon_edit_distance, out_lexicon_edit_distance = \
                self.gather_edit_distance([(m['edit_distance'], m['in_lexicon']) for m in raw_metrics])
            total_accuracy, in_lexicon_accuracy, out_lexicon_accuracy = \
                self.gather_edit_distance([(m['accuracy'], m['in_lexicon']) for m in raw_metrics])
            return dict(total_edit_distance=total_edit_distance, in_lexicon_edit_distance=in_lexicon_edit_distance,
                        out_lexicon_edit_distance=out_lexicon_edit_distance, total_accuracy=total_accuracy,
                        in_lexicon_accuracy=in_lexicon_accuracy, out_lexicon_accuracy=out_lexicon_accuracy)

    def gather_accuracy(self, raw_metrics):
        accuracy_meter = AverageMeter()
        for raw_metric in raw_metrics:
            total = len(raw_metric)
            accuracy = np.array(raw_metric).sum() / total
            accuracy_meter.update(accuracy, total)
        return accuracy_meter

    def in_lexicon(self, batch, output):
        result = []
        for i in range(len(output)):
            label_string = output[i]['label_string'].upper()
            result.append(label_string in self.nori_lexicon)
        return result

    def accuracy(self, batch, output):
        eq = []
        for i in range(len(output)):
            label_string = output[i]['label_string'].upper()
            pred_string = output[i]['pred_string'].upper()
            eq.append(label_string == pred_string)
        return eq

    def gather_edit_distance(self, raw_metrics_ed, logger: Logger = None):
        meter = AverageMeter()
        if not self.nori_lexicon:
            for raw_metric in raw_metrics_ed:
                total = len(raw_metric)
                edit_distance = np.array(raw_metric).sum() / total
                meter.update(edit_distance, total)
            return meter
        else:
            in_lexicon_meter = AverageMeter()
            out_lexicon_meter = AverageMeter()
            for raw_metric, in_lexicon in raw_metrics_ed:
                raw_metric = np.array(raw_metric)
                in_lexicon = np.array(in_lexicon)

                total = len(raw_metric)
                edit_distance = raw_metric.sum() / total
                meter.update(edit_distance, total)

                all_in_lexicon = raw_metric[in_lexicon == True]
                in_lexicon_edit_distance = all_in_lexicon.sum() / max(len(all_in_lexicon), 1)
                in_lexicon_meter.update(in_lexicon_edit_distance, len(all_in_lexicon))

                all_out_lexicon = raw_metric[in_lexicon == False]
                out_lexicon_edit_distance = all_out_lexicon.sum() / max(len(all_out_lexicon), 1)
                out_lexicon_meter.update(out_lexicon_edit_distance, len(all_out_lexicon))
            return meter, in_lexicon_meter, out_lexicon_meter

    def edit_distance(self, batch, output):
        ed_vec = []
        for i in range(len(output)):
            label_string = output[i]['label_string'].upper()
            pred_string = output[i]['pred_string'].upper()
            length = len(label_string)
            if length == 0:
                ed_vec.append(0.0)
            else:
                ed_vec.append(float(1 - min(length, ed.eval(label_string, pred_string)) * 1.0 / length))
        return ed_vec
