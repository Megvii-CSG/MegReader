import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

from concern.config import Configurable, State
import concern.webcv2 as webcv2


class SegDetectorRepresenter(Configurable):
    thresh = State(default=0.3)
    box_thresh = State(default=0.7)
    max_candidates = State(default=100)
    resize = State(default=False)

    dest = State(default='binary')

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)

        self.min_size = 3
        self.scale_ratio = 0.4
        if 'debug' in cmd:
            self.debug = cmd['debug']
        if 'thresh' in cmd:
            self.thresh = cmd['thresh']
        if 'box_thresh' in cmd:
            self.box_thresh = cmd['box_thresh']
        if 'dest' in cmd:
            self.dest = cmd['dest']

    def represent(self, batch, _pred):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, 1, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, 1, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, 1, H, W)
        '''
        images = batch['image']
        pred = _pred[self.dest]
        segmentation = self.binarize(pred)
        boxes_batch = []
        preds = []
        for batch_index in range(images.size(0)):
            height, width = batch['shape'][batch_index]
            boxes, single_pred = self.boxes_from_bitmap(
                _pred['binary'][batch_index],
                segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            preds.append(single_pred.reshape(1, *single_pred.shape))
        return boxes_batch, _pred

    def binarize(self, pred):
        return pred > self.thresh

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        assert _bitmap.size(0) == 1
        bitmap = _bitmap.data.cpu().numpy()[0]  # The first channel
        pred = pred.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        boxes = []
        _, contours, _ = cv2.findContours(
            (bitmap*255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if self.debug:
            bitmap = cv2.cvtColor(pred * 255, cv2.COLOR_GRAY2BGR)

        for contour in contours[:self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))

            if self.debug:
                points = points.astype(np.int32)
                bitmap = cv2.polylines(
                        bitmap, [points.reshape(-1, 2)], True, (255, 0, 0), 3)
                bitmap = cv2.putText(
                        bitmap, str(round(score, 3)),
                        (points[:, 0].min(), points[:, 1].min()),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            if self.box_thresh > score:
                continue
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            if not self.resize:
                dest_width = width
                dest_height = height

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())

        if self.debug:
            webcv2.imshow('mask', bitmap)
        return boxes, bitmap

    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * 1.5 / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
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

    def box_score(self, bitmap, box):
        '''
        naive version of box score computation,
        only for helping principle understand.
        '''
        mask = np.zeros_like(bitmap, dtype=np.uint8)
        cv2.fillPoly(mask, box.reshape(1, 4, 2).astype(np.int32), 1)
        return cv2.mean(bitmap, mask)[0]

    def box_score_fast(self, bitmap, _box):
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
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]
