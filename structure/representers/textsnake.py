import pickle

import cv2
import numpy as np

from concern.convert import to_np
from concern.textsnake import fill_hole, regularize_sin_cos, norm2, vector_sin, vector_cos
from concern.config import Configurable, State


class TextsnakeResultBuilder(Configurable):
    tr_thresh = State(default=0.6)
    tcl_thresh = State(default=0.4)

    @staticmethod
    def find_innerpoint(cont):
        """
        generate an inner point of input polygon using mean of x coordinate by:
        1. calculate mean of x coordinate(xmean)
        2. calculate maximum and minimum of y coordinate(ymax, ymin)
        2. iterate for each y in range (ymin, ymax), find first segment in the polygon
        3. calculate means of segment
        :param cont: input polygon
        :return:
        """

        xmean = cont[:, 0, 0].mean()
        ymin, ymax = cont[:, 0, 1].min(), cont[:, 0, 1].max()
        found = False
        found_y = []
        for i in np.arange(ymin - 1, ymax + 1, 0.5):
            # if in_poly > 0, (xmean, i) is in `cont`
            in_poly = cv2.pointPolygonTest(cont, (xmean, i), False)
            if in_poly > 0:
                found = True
                found_y.append(i)
            # first segment found
            if in_poly < 0 and found:
                break

        if found_y:
            return xmean, np.array(found_y).mean()

        # if cannot find using above method, try each point's neighbor
        else:
            for p in range(len(cont)):
                point = cont[p, 0]
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        test_pt = point + [i, j]
                        if cv2.pointPolygonTest(cont, (test_pt[0], test_pt[1]), False) > 0:
                            return test_pt

    def in_contour(self, cont, point):
        x, y = point
        return cv2.pointPolygonTest(cont, (x, y), False) >= 0

    def centerlize(self, x, y, height, width, tangent_cos, tangent_sin, tcl_contour, stride=1):
        """
        centralizing (x, y) using tangent line and normal line.
        :return:
        """

        # calculate normal sin and cos
        normal_cos = -tangent_sin
        normal_sin = tangent_cos

        # find upward
        _x, _y = x, y
        while self.in_contour(tcl_contour, (_x, _y)):
            _x = _x + normal_cos * stride
            _y = _y + normal_sin * stride
            if int(_x) >= width or int(_x) < 0 or int(_y) >= height or int(_y) < 0:
                break
        end1 = np.array([_x, _y])

        # find downward
        _x, _y = x, y
        while self.in_contour(tcl_contour, (_x, _y)):
            _x = _x - normal_cos * stride
            _y = _y - normal_sin * stride
            if int(_x) >= width or int(_x) < 0 or int(_y) >= height or int(_y) < 0:
                break
        end2 = np.array([_x, _y])

        # centralizing
        center = (end1 + end2) / 2
        return center

    def mask_to_tcl(self, pred_sin, pred_cos, pred_radius, tcl_contour, init_xy, direct=1):
        """
        Iteratively find center line in tcl mask using initial point (x, y)
        :param pred_sin: predict sin map
        :param pred_cos: predict cos map
        :param tcl_mask: predict tcl mask
        :param init_xy: initial (x, y)
        :param direct: direction [-1|1]
        :return:
        """

        height, width = pred_sin.shape
        x_init, y_init = init_xy

        sin = pred_sin[int(y_init), int(x_init)]
        cos = pred_cos[int(y_init), int(x_init)]
        radius = pred_radius[int(y_init), int(x_init)]

        x_shift, y_shift = self.centerlize(x_init, y_init, height, width, cos, sin, tcl_contour)
        result = []

        attempt = 0
        while self.in_contour(tcl_contour, (x_shift, y_shift)):

            if attempt < 100:
                attempt += 1
            else:
                break

            x, y = x_shift, y_shift

            sin = pred_sin[int(y), int(x)]
            cos = pred_cos[int(y), int(x)]

            x_c, y_c = self.centerlize(x, y, height, width, cos, sin, tcl_contour)
            result.append(np.array([x_c, y_c, radius]))

            sin_c = pred_sin[int(y_c), int(x_c)]
            cos_c = pred_cos[int(y_c), int(x_c)]
            radius = pred_radius[int(y_c), int(x_c)]

            # shift stride
            for shrink in [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]:
                # stride = +/- 0.5 * [sin|cos](theta), if new point is outside, shrink it until shrink < 0.1, hit ends
                t = shrink * radius
                x_shift_pos = x_c + cos_c * t * direct  # positive direction
                y_shift_pos = y_c + sin_c * t * direct  # positive direction
                x_shift_neg = x_c - cos_c * t * direct  # negative direction
                y_shift_neg = y_c - sin_c * t * direct  # negative direction

                # if first point, select positive direction shift
                if len(result) == 1:
                    x_shift, y_shift = x_shift_pos, y_shift_pos
                else:
                    # else select point further with second last point
                    dist_pos = norm2(result[-2][:2] - (x_shift_pos, y_shift_pos))
                    dist_neg = norm2(result[-2][:2] - (x_shift_neg, y_shift_neg))
                    if dist_pos > dist_neg:
                        x_shift, y_shift = x_shift_pos, y_shift_pos
                    else:
                        x_shift, y_shift = x_shift_neg, y_shift_neg
                # if out of bounds, skip
                if int(x_shift) >= width or int(x_shift) < 0 or int(y_shift) >= height or int(y_shift) < 0:
                    continue
                # found an inside point
                if self.in_contour(tcl_contour, (x_shift, y_shift)):
                    break
            # if out of bounds, break
            if int(x_shift) >= width or int(x_shift) < 0 or int(y_shift) >= height or int(y_shift) < 0:
                break
        return result

    def build_tcl(self, tcl_pred, sin_pred, cos_pred, radius_pred):
        """
        Find TCL's center points and radius of each point
        :param tcl_pred: output tcl mask, (512, 512)
        :param sin_pred: output sin map, (512, 512)
        :param cos_pred: output cos map, (512, 512)
        :param radius_pred: output radius map, (512, 512)
        :return: (list), tcl array: (n, 3), 3 denotes (x, y, radius)
        """
        all_tcls = []

        # find disjoint regions
        mask = fill_hole(tcl_pred)
        _, conts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        conts = [cont for cont in conts if cv2.contourArea(cont) > 20]

        for index, cont in enumerate(conts):

            # find an inner point of polygon
            init = self.find_innerpoint(cont)

            if init is None:
                continue

            x_init, y_init = init

            # find left tcl
            tcl_left = self.mask_to_tcl(sin_pred, cos_pred, radius_pred, cont, (x_init, y_init), direct=1)
            tcl_left = np.array(tcl_left)
            # find right tcl
            tcl_right = self.mask_to_tcl(sin_pred, cos_pred, radius_pred, cont, (x_init, y_init), direct=-1)
            tcl_right = np.array(tcl_right)
            # concat
            tcl = np.concatenate([tcl_left[::-1][:-1], tcl_right])
            if len(tcl) > 1:
                all_tcls.append(tcl)

        return all_tcls

    def detect(self, tr_pred, tcl_pred, sin_pred, cos_pred, radius_pred):
        tr_pred_mask = tr_pred > self.tr_thresh
        tcl_pred_mask = tcl_pred > self.tcl_thresh

        tcl = tcl_pred_mask * tr_pred_mask

        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

        detect_result = self.build_tcl(tcl, sin_pred, cos_pred, radius_pred)

        return detect_result

    def result2polygon(self, mask_shape, result):
        """ convert geometric info(center_x, center_y, radius) into contours
        :param image: image
        :param result: (list), each with (n, 3), 3 denotes (x, y, radius)
        :return: (np.ndarray list), polygon format contours
        """
        all_conts = []
        for disk in result:
            mask = np.zeros(mask_shape, dtype=np.uint8)
            for x, y, r in disk:
                if r > 1:
                    cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)

            _, conts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(conts) > 1:
                conts.sort(key=lambda x: cv2.contourArea(x), reverse=True)
            elif not conts:
                continue
            if cv2.contourArea(conts[0]) < 20:
                continue
            all_conts.append(conts[0][:, 0, :])
        return all_conts

    def remove_fp_1(self, tcl_pred_mask, detect_tcl, contours):
        result = []
        for contour, disk in zip(contours, detect_tcl):
            mask = np.zeros(tcl_pred_mask.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 1)
            tcl_sum = (mask * tcl_pred_mask).sum()
            area = 0
            for i in range(len(disk) - 1):
                area += (disk[i][2] + disk[i + 1][2]) / 2 * norm2(
                    disk[i][:2] - disk[i + 1][:2]) * 0.2
            if tcl_sum > area:
                result.append(contour)
        return result

    def remove_fp_2(self, tr_pred_mask, contours):
        result = []
        for contour in contours:
            mask = np.zeros(tr_pred_mask.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 1)
            if (tr_pred_mask * mask).sum() * 2 > mask.sum():
                result.append(contour)
        return result

    def stride(self, detect_tcl, contours, detected_conts, left=True, step=0.5):
        if left:
            last_point, before_point = detect_tcl[:2]
        else:
            before_point, last_point = detect_tcl[-2:]
        radius = last_point[2]
        cos = vector_cos(last_point[:2] - before_point[:2])
        sin = vector_sin(last_point[:2] - before_point[:2])
        new_point = last_point[:2] + radius * step * np.array((cos, sin))
        for index, contour in enumerate(contours):
            if index not in detected_conts and (self.in_contour(contour, new_point)):
                return new_point, index
        return None, None

    def remove_fp_stride(self, mask_shape, detect_tcl, contours):
        detected = set()
        result = []
        for index, tcl in enumerate(detect_tcl):
            if index in detected:
                continue
            detected.add(index)
            _, left_index = self.stride(tcl, contours, detected, step=0)
            _, right_index = self.stride(tcl, contours, detected, False, step=0)
            if left_index:
                detected.add(left_index)
                tcl = np.concatenate((tcl, detect_tcl[left_index]))
            if right_index:
                detected.add(right_index)
                tcl = np.concatenate((tcl, detect_tcl[right_index]))
            result.append(tcl)
        return self.result2polygon(mask_shape, result)

    def get_polygons(self, tr_pred, tcl_pred, sin_pred, cos_pred, radius_pred):
        tr_pred_mask = tr_pred > self.tr_thresh
        result = self.detect(tr_pred, tcl_pred, sin_pred, cos_pred, radius_pred)
        polygons = self.result2polygon(sin_pred.shape, result)
        polygons = self.remove_fp_stride(sin_pred.shape, result, polygons)
        polygons = self.remove_fp_2(tr_pred_mask, polygons)
        return polygons


class TextsnakeRepresenter(Configurable):
    result_builder = TextsnakeResultBuilder()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def represent_batch(self, batch):
        image, label, meta = batch
        batch_size = image.shape[0]

        output = {
            'image': to_np(image).transpose(0, 2, 3, 1),
            **{key: to_np(value) for key, value in label.items()},
            'meta': [pickle.loads(value) for value in meta],
        }

        output['contours_gt'] = [self.result_builder.get_polygons(
            output['tr_mask'][i],
            output['tcl_mask'][i],
            output['sin_map'][i],
            output['cos_map'][i],
            output['radius_map'][i],
        ) for i in range(batch_size)]

        output['polygons_gt'] = [output['meta'][i]['Cnts'] for i in range(batch_size)]

        return output

    def represent(self, batch, pred):
        image, label, meta = batch
        batch_size = image.shape[0]

        output = self.represent_batch(batch)
        output.update({key: to_np(value) for key, value in pred.items()})
        output['contours_pred'] = [self.result_builder.get_polygons(
            output['tr_pred'][i],
            output['tcl_pred'][i],
            output['sin_pred'][i],
            output['cos_pred'][i],
            output['radius_pred'][i],
        ) for i in range(batch_size)]

        return output
