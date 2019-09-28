from concern.config import State
from .sequence_recognition_representer import SequenceRecognitionRepresenter
import config

import cv2
import numpy as np


class SegRecognitionRepresenter(SequenceRecognitionRepresenter):
    max_candidates = State(default=64)
    min_size = State(default=2)
    thresh = State(default=0.3)
    box_thresh = State(default=0.7)

    def represent(self, batch, pred):
        labels = batch['label']
        mask = pred['mask']
        classify = pred['classify']
        result = []

        for batch_index in range(mask.shape[0]):
            label_string = self.label_to_string(labels[batch_index])
            result_dict = self.result_from_heatmap(mask[batch_index], classify[batch_index])
            result_dict.update(label_string=label_string)
            result.append(result_dict)
        return result

    def result_from_heatmap(self, bitmap, heatmap):
        bitmap = (bitmap > self.thresh).data.cpu().numpy()
        score_map = heatmap.data.cpu().detach().numpy()
        result = []
        _, contours, _ = cv2.findContours(
            (bitmap*255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        boxes = []
        for contour in contours[:self.max_candidates]:
            points, sside = self.get_boxes(contour)

            #if sside < self.min_size:
            #    print('by min side')
            #    continue
            boxes.append(np.array(points).reshape(-1, 2))

        for points in sorted(boxes, key=lambda x: x[0][0]):
            score, char_index = self.box_score_fast(bitmap, points.reshape(-1, 2), score_map)
            #if self.box_thresh > score:
            #    continue
            result.append(char_index)

        pred_string = self.charset.label_to_string(result)
        return dict(mask=bitmap, classify=score_map, pred_string=pred_string)

    def get_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box, score_map):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        score = cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]
        char_index = int(score_map[1:, ymin:ymax+1, xmin:xmax+1].mean(axis=1).mean(axis=1).argmax()) + 1
        return score, char_index
