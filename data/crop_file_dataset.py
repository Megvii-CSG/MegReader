import torch
import torch.utils.data as data
import cv2
import numpy as np
import glob
import os
from concern.config import Configurable, State
from concern.cv import min_area_rect

import config
from concern.charsets import DefaultCharset
from data.processes.resize_image import ResizeImage


class CropFileDataset(data.Dataset, Configurable):
    file_pattern = State()
    charset = State(default=DefaultCharset())
    max_size = State(default=32)
    image_size = State(default=[64, 512])
    mode = State(default='resize')

    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

    def __init__(self, file_pattern=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.file_pattern = file_pattern or self.file_pattern
        self.resize = ResizeImage(self.image_size, self.mode)
        self.prepare()

    def prepare(self):
        self.file_paths = glob.glob(self.file_pattern)
        assert len(self.file_paths) > 0

        self.gt = []
        for file_name in self.file_paths:
            base_name = os.path.basename(file_name)
            names = base_name.split('_')
            gt_with_suffix = '_'.join(names[1:])
            assert gt_with_suffix.endswith('.jpg')
            gt = gt_with_suffix[:gt_with_suffix.rindex('.jpg')]
            self.gt.append(gt)
        self.num_samples = len(self.file_paths)
        return self

    def is_vertival(self, height, width):
        return height > width * 1.5

    def ensure_horizontal(self, image):
        if self.is_vertival(*image.shape[:2]):
            image = np.flip(np.swapaxes(image, 0, 1), 0)
        return image

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        file_path = self.file_paths[index]
        gt = self.gt[index]

        image = cv2.imread(file_path, cv2.IMREAD_COLOR).astype('float32')
        image = self.ensure_horizontal(image)
        image = self.resize(image)

        image -= self.RGB_MEAN
        image /= 255.
        length = np.array(min(len(gt), self.max_size), dtype=np.int32)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = self.gt_to_label(gt, image)
        return image, label, length

    def gt_to_label(self, gt, image=None):
        return self.charset.string_to_label(gt)[:config.max_size]

    def __len__(self):
        return self.num_samples

    @classmethod
    def restore(cls, data):
        data = data.permute(1, 2, 0).to('cpu').data.numpy()
        data = data * 255.
        data += cls.RGB_MEAN
        return data.astype(np.uint8)


class ImageCropper(Configurable):
    charset = State(default=DefaultCharset())
    max_size = State(default=32)
    image_size = State(default=[64, 512])
    mode = State(default='resize')

    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.resize = ResizeImage(self.image_size, self.mode)

    def is_vertival(self, height, width):
        return height > width * 1.5

    def ensure_horizontal(self, image):
        if self.is_vertival(*image.shape[:2]):
            image = np.flip(np.swapaxes(image, 0, 1), 0)
        return image

    def crop(self, image, poly):
        box = min_area_rect(poly)

        w = np.linalg.norm(box[1] - box[0])
        h = np.linalg.norm(box[2] - box[1])

        src = box.astype('float32')
        dst = np.array([(0, 0), (w, 0), (w, h), (0, h)], 'float32')
        mat = cv2.getPerspectiveTransform(src, dst)

        image = cv2.warpPerspective(image, mat, (w, h))
        image = image.astype('float32')

        image = self.ensure_horizontal(image)
        image = self.resize(image)

        image -= self.RGB_MEAN
        image /= 255.

        return image
