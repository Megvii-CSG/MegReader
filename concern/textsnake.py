import cv2
import numpy as np


def fill_hole(input_mask):
    height, width = input_mask.shape
    canvas = np.zeros((height + 2, width + 2), np.uint8)
    canvas[1:height + 1, 1:width + 1] = input_mask.copy()

    mask = np.zeros((height + 4, width + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:height + 1, 1:width + 1].astype(np.bool)

    return ~canvas | input_mask.astype(np.uint8)


def regularize_sin_cos(sin, cos):
    # regularization
    scale = np.sqrt(1.0 / (sin ** 2 + cos ** 2))
    return sin * scale, cos * scale


def norm2(x, axis=None):
    return np.sqrt(np.sum(x ** 2, axis=axis))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))


def vector_sin(v):
    assert len(v) == 2
    return v[1] / np.sqrt(v[0] ** 2 + v[1] ** 2)


def vector_cos(v):
    assert len(v) == 2
    return v[0] / np.sqrt(v[0] ** 2 + v[1] ** 2)


def find_bottom(pts):
    pts = np.array(pts)
    if len(pts) > 4:
        e = np.concatenate([pts, pts[:3]])
        candidate = []
        for i in range(1, len(pts) + 1):
            v_prev = e[i] - e[i - 1]
            v_next = e[i + 2] - e[i + 1]
            if cos(v_prev, v_next) < -0.7:
                candidate.append((i % len(pts), (i + 1) % len(pts), cos(v_prev, v_next)))
        candidate.sort(key=lambda x: x[2])

        if len(candidate) < 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
            # if candidate number < 2, or two bottom are joined, select 2 farthest edge
            mid_list = []
            for i in range(len(pts)):
                mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                mid_list.append((i, (i + 1) % len(pts), mid_point))

            dist_list = []
            for i in range(len(pts)):
                for j in range(len(pts)):
                    s1, e1, mid1 = mid_list[i]
                    s2, e2, mid2 = mid_list[j]
                    dist = norm2(mid1 - mid2)
                    dist_list.append((s1, e1, s2, e2, dist))
            bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
            bottoms = [dist_list[bottom_idx[0]][:2], dist_list[bottom_idx[1]][:2]]
        else:
            bottoms = [candidate[0][:2], candidate[1][:2]]
        if bottoms[0][0] == bottoms[1][1] or bottoms[1][0] == bottoms[0][1]:
            print(1)

    else:
        # try:
        d1 = norm2(pts[1] - pts[0]) + norm2(pts[2] - pts[3])
        d2 = norm2(pts[2] - pts[1]) + norm2(pts[0] - pts[3])
        bottoms = [(0, 1), (2, 3)] if d1 < d2 else [(1, 2), (3, 0)]
        # except:
        #     print(pts)
    assert len(bottoms) == 2, 'fewer than 2 bottoms'
    return bottoms


def split_long_edges(points, bottoms):
    """
    Find two long edge sequence of and polygon
    """
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)

    i = b1_end + 1
    long_edge_1 = []
    while i % n_pts != b2_end:
        long_edge_1.append((i - 1, i))
        i = (i + 1) % n_pts

    i = b2_end + 1
    long_edge_2 = []
    while i % n_pts != b1_end:
        long_edge_2.append((i - 1, i))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def find_long_edges(points, bottoms):
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)
    i = b1_end
    long_edge_1 = []

    while i != b2_start:
        start = i
        end = (i + 1) % n_pts
        long_edge_1.append((start, end))
        i = end

    i = b2_end % n_pts
    long_edge_2 = []
    while i != b1_start:
        start = i
        end = (i + 1) % n_pts
        long_edge_2.append((start, end))
        i = end
    return long_edge_1, long_edge_2


def split_edge_seqence(points, long_edge, n_parts):
    points = np.array(points)

    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = point_cumsum[-1]
    length_per_part = total_length / n_parts

    # first point
    cur_node = 0
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        try:
            while cur_end > point_cumsum[cur_node + 1]:
                cur_node += 1
        except IndexError:
            print(points)

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)
