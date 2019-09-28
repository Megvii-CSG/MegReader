import numpy as np
import scipy.ndimage.filters as fi

from concern.config import State

from .data_process import DataProcess


class MakeDecoupleMap(DataProcess):
    max_size = State(default=32)
    shape = State(default=(64, 256))
    sigma = State(default=2)
    summation = State(default=False)
    box_key = State(default='charboxes')
    function = State(default='gaussian')

    def process(self, data):
        assert self.box_key in data, '%s in data is required' % self.box_key
        shape = data['image'].shape[:2]
        boxes = np.array(data[self.box_key])

        ratio_x = shape[1] / self.shape[1]
        boxes[:, :, 0] = (boxes[:, :, 0] / ratio_x).clip(0, self.shape[1])
        ratio_y = shape[0] / self.shape[0]
        boxes[:, :, 1] = (boxes[:, :, 1] / ratio_y).clip(0, self.shape[0])
        boxes = (boxes + .5).astype(np.int32)
        xmins = boxes[:, :, 0].min(axis=1)
        xmaxs = np.maximum(boxes[:, :, 0].max(axis=1), xmins + 1)

        ymins = boxes[:, :, 1].min(axis=1)
        ymaxs = np.maximum(boxes[:, :, 1].max(axis=1), ymins + 1)

        if self.summation:
            canvas = np.zeros(self.shape, dtype=np.int32)
        else:
            canvas = np.zeros((self.max_size, *self.shape), dtype=np.float32)

        mask = np.zeros(self.shape, dtype=np.float32)
        orders = self.orders(data)
        for i in range(xmins.shape[0]):
            function = getattr(self, 'render_' + self.function)
            order = min(orders[i], self.max_size)
            if self.summation:
                function(canvas, xmins[i], xmaxs[i], ymins[i], ymaxs[i],
                         value=order+1, shrink=0.6)
            else:
                function(canvas[order], xmins[i], xmaxs[i], ymins[i], ymaxs[i])
            self.render_gaussian(mask, xmins[i], xmaxs[i], ymins[i], ymaxs[i])

        data['ordermaps'] = canvas
        data['mask'] = mask
        return data

    def orders(self, data):
        orders = []
        if 'lines' in data:
            for text in data['lines'].texts:
                orders += list(range(len(text)))
        else:
            orders = list(range(data[self.box_key].shape[0]))
        return orders

    def render_gaussian_thresh(self, canvas, xmin, xmax, ymin, ymax,
                               value=1, thresh=0.2, shrink=None):
        out = np.zeros_like(canvas).astype(np.float32)
        out[(ymax+ymin+1)//2, (xmax+xmin+1)//2] = 1.
        h, w = canvas.shape[:2]
        out = fi.gaussian_filter(out, (self.sigma, self.sigma),
                           output=out, mode='mirror')
        out = out / out.max()
        canvas[out > thresh] = value

    def render_gaussian(self, canvas, xmin, xmax, ymin, ymax):
        out = np.zeros_like(canvas)
        out[(ymax+ymin+1)//2, (xmax+xmin+1)//2] = 1.
        h, w = canvas.shape[:2]
        fi.gaussian_filter(out, (self.sigma, self.sigma),
                           output=out, mode='mirror')
        out = out / out.max()
        canvas[out > canvas] = out[out > canvas]

    def render_gaussian_fast(self, canvas, xmin, xmax, ymin, ymax):
        out = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.float32)
        out[(ymax-ymin+1)//2, (xmax-xmin+1)//2] = 1.
        h, w = canvas.shape[:2]
        fi.gaussian_filter(out, (self.sigma, self.sigma),
                           output=out, mode='mirror')
        out = out / out.max()
        canvas[ymin:ymax+1, xmin:xmax+1] = np.maximum(out, canvas[ymin:ymax+1, xmin:xmax+1])
