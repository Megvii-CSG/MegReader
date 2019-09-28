import torch
import cv2
import numpy as np

from concern.config import Configurable, State


class TextsnakeVisualizer(Configurable):
    vis_num = State(default=4)

    def __init__(self, **kwargs):
        pass

    def gt_get_image(self, idx, output):
        img = output['image'][idx]
        img_show = img.copy()

        tr_mask = output['tr_mask'][idx]
        tcl_mask = output['tcl_mask'][idx]

        polygons_gt = output['polygons_gt'][idx]
        contours_gt = output['contours_gt'][idx]

        contours_gt_im = np.zeros_like(img_show)
        for contour in contours_gt:
            cv2.fillPoly(
                contours_gt_im, np.array([contour], 'int32'),
                (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            )

        gt_vis = self.visualize_detection(img_show, tr_mask, tcl_mask, polygons_gt)
        gt_vis = np.concatenate([gt_vis, contours_gt_im], axis=1)

        return gt_vis

    def pred_get_image(self, idx, output):
        img = output['image'][idx]

        tr_pred = output['tr_pred'][idx]
        tcl_pred = output['tcl_pred'][idx]

        # visualization
        img_show = img.copy()
        contours = output['contours_pred'][idx]
        contours_im = np.zeros_like(img_show)
        for contour in contours:
            cv2.fillPoly(
                contours_im, np.array([contour], 'int32'),
                (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            )

        predict_vis = self.visualize_detection(img_show, tr_pred, tcl_pred, contours)
        predict_vis = np.concatenate([predict_vis, contours_im], axis=1)

        return predict_vis

    def get_image(self, idx, output):
        gt_vis = self.gt_get_image(idx, output)
        predict_vis = self.pred_get_image(idx, output)
        return np.concatenate([predict_vis, gt_vis], axis=0)

    def visualize_detection(self, image, tr, tcl, contours):
        image_show = image.copy()
        for contour in contours:
            image_show = cv2.polylines(image_show, np.array([contour], 'int32'), True, (0, 0, 255), 2)
        tr = cv2.cvtColor((tr * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        tcl = cv2.cvtColor((tcl * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        image_show = np.concatenate([image_show, tr, tcl], axis=1)
        return image_show

    def visualize(self, batch, output, interested):
        images = {}
        for i in range(min(self.vis_num, len(output['image']))):
            show = self.get_image(i, output)
            images['image_%d' % i] = show.astype(np.uint8)
        return images

    def visualize_batch(self, batch, output):
        images = {}
        for i in range(len(output['image'])):
            show = self.gt_get_image(i, output)
            train_mask = cv2.cvtColor((output['train_mask'][i] * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
            show = np.concatenate([show, train_mask], axis=1)
            images['image_%d' % i] = show.astype(np.uint8)
        return images
