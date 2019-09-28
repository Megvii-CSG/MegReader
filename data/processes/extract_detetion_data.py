import imgaug
from concern.config import Configurable, State
from data.augmenter import AugmenterBuilder


class ExtractDetectionData(Configurable):
    augmenter_args = State(autoload=False)
    augmenter_class = State(default=None)

    def __init__(self, augmenter_class=None, **kwargs):
        self.augmenter_args = kwargs.get('augmenter_args')
        self.augmenter_class = augmenter_class or self.augmenter_class
        if self.augmenter_class is not None:
            self.augmenter = eval(self.augmenter_class)().augmenter
        else:
            self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_poly(self, aug, img_shape, poly):
        if aug is None:
            return poly
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

    def __call__(self, data):
        img = data['img']
        aug = None
        shape = img.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            img = aug.augment_image(data['img'])

        line_polys = []
        for line in data['lines']:
            line_polys.append({
                'points': self.may_augment_poly(aug, shape, line['poly']),
                'ignore': line['text'] == '###',
                'text': line['text'],
            })
        filename = data.get('filename', data.get('data_id', ''))
        label = {
            'polys': line_polys
        }
        meta = {
            'data_id': data['data_id'],
            'filename': filename,
            'shape': shape[:2]
        }
        return img, label, meta
