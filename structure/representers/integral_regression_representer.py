import numpy as np
import torch
from concern.config import State
from .sequence_recognition_representer import SequenceRecognitionRepresenter


class IntegralRegressionRepresenter(SequenceRecognitionRepresenter):
    represent_type = State(default='max')
    order_th = State(default=0.1)
    loc_th = State(default=0.1)
    score_th = State(default=0.3)
    vis = State(default=True)

    def represent(self, batch, pred):
        labels = batch['label']
        global_map = (pred['global_map'] > self.loc_th).float() * pred['global_map']
        heatmap = (pred['ordermaps'][:, 1:] > self.order_th).float() * global_map
        # heatmap = pred['heatmap']*(pred['heatmap']>0.2).float()
        N, C, H, W = heatmap.shape
        score = heatmap / torch.max(heatmap.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True), torch.tensor(1e-8))
        if self.represent_type == 'integral':
            index_y, index_x = torch.meshgrid(torch.linspace(0, H - 1, H),
                                              torch.linspace(0, W - 1, W))
            index_x = index_x.repeat(N, C, 1, 1)  # N, H, W
            index_y = index_y.repeat(N, C, 1, 1)  # N, H, W
            max_x = (score * index_x).sum(2).sum(2).data.cpu().numpy()  # N, C
            max_x = (max_x + 0.5).astype(np.int32)
            max_y = (score * index_y).sum(2).sum(2).data.cpu().numpy()  # N, C
            max_y = (max_y + 0.5).astype(np.int32)
            result = self.gen_result(pred['classify'], labels, max_x, max_y, heatmap)
        elif self.represent_type == 'max':
            max_x = heatmap.max(dim=2)[0].argmax(dim=2).data.cpu().numpy()  # N, C
            max_y = heatmap.max(dim=3)[0].argmax(dim=2).data.cpu().numpy()  # N, C
            result = self.gen_result(pred['classify'], labels, max_x, max_y, heatmap)
        elif self.represent_type == 'weighted_max':
            # heatmap = (heatmap > self.thresh).float() * heatmap
            result = self.gen_result_weighted(pred['classify'], heatmap, labels)
        else:
            raise NotImplementedError
        if 'global_map' in pred.keys():
            for ind, res in enumerate(result):
                res.update(dict(global_map=pred['global_map'].data.cpu().numpy()))
        if 'ordermaps' in pred.keys():
            for ind, res in enumerate(result):
                res.update(dict(ordermaps=pred['ordermaps'].data.cpu().numpy()))
        if self.vis:
            pass

        return result

    def keep_class_max(self, maps):
        val, inds = maps.max(1)
        val_map = val.unsqueeze(1).expand_as(maps)
        out = torch.where(maps == val_map, maps, torch.tensor(0.))
        return out

    def gen_result(self, pred_classify, labels, max_x, max_y, heatmap):
        classify = pred_classify.argmax(dim=1).data.cpu().numpy()  # N, H, W
        result = []
        for batch_index in range(classify.shape[0]):
            label_string = self.label_to_string(labels[batch_index])
            pred_string = self.collect_chars(
                pred_classify[batch_index], max_x[batch_index], max_y[batch_index], heatmap[batch_index])
            result.append(dict(label_string=label_string, pred_string=pred_string,
                               heatmap=heatmap, classify=pred_classify, x=max_x[batch_index], y=max_y[batch_index]))
        return result

    def gen_result_weighted(self, classify, heatmap, labels):
        result = []
        for batch_index in range(classify.shape[0]):
            label_string = self.label_to_string(labels[batch_index])
            p_label = []
            for ind, h_map in enumerate(heatmap[batch_index]):
                c_map = classify[batch_index] * h_map
                c_score = c_map.sum(1).sum(1)
                if c_score.max() > self.score_th:
                    c_char = c_score.argmax().item()
                    p_label.append(c_char)
                else:
                    break
            pred_string = self.label_to_string(p_label)
            result.append(dict(label_string=label_string, pred_string=pred_string,
                               heatmap=heatmap, classify=classify))
        return result

    def collect_chars(self, classify, x, y, heatmap):
        result = []
        for char_index in range(x.shape[0]):
            if heatmap[char_index].max() < self.score_th:
                klass = 0
            else:
                klass = classify[:, y[char_index]-1:y[char_index]+1, x[char_index]-1:x[char_index]+1].mean(1).mean(1).argmax().item()
            result.append(klass)
        return self.label_to_string(result)
