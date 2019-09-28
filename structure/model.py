import glob
import os
import pickle

import numpy as np
import apex

import torch
import torch.nn as nn
import torch.nn.functional as F

import backbones
import decoders


class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        # return nn.parallel.DistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[local_rank],
        #     output_device=[local_rank],
        #     find_unused_parameters=True)
        return apex.parallel.DistributedDataParallel(model.cuda())
    else:
        return nn.DataParallel(model)


class ClassificationModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        nn.Module.__init__(self)

        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return args['backbone'] + '-' + args['decoder']

    def forward(self, batch, training=True):
        data, label = batch
        data = data.to(self.device)
        label = label.to(self.device)

        if training:
            loss, pred = self.model(data, targets=label, train=True)
            return loss, pred
        else:
            return self.model(data, train=False)


class DetectionModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        nn.Module.__init__(self)

        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return args['backbone'] + '-' + args['decoder']

    def forward(self, batch, training=True):
        data, label, meta = batch
        data = data.to(self.device)
        data = data.float() / 255.0
        for key, value in label.items():
            label[key] = value.to(self.device)

        if training:
            loss, pred, metrics = self.model(data, label, meta, train=True)
            loss = loss.mean()
            return loss, pred, metrics
        else:
            return self.model(data, label, meta, train=False)


class DetectionEnsembleModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        nn.Module.__init__(self)

        self.sizes = args['sizes']
        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return args['backbone'] + '-' + args['decoder']

    def forward(self, batch, training=True):
        assert (not training)
        data, label, meta = batch
        data = data.to(self.device)
        data = data.float() / 255.0
        for key, value in label.items():
            label[key] = value.to(self.device)

        size = (data.shape[2], data.shape[3])

        heatmaps = []
        for size0 in self.sizes:
            data0 = F.interpolate(data, size0, mode='bilinear')
            pred0 = self.model(data0, label, meta, train=False)
            heatmap0 = F.interpolate(pred0['heatmap'], size, mode='bilinear')
            heatmaps.append(heatmap0)
        heatmap = sum(heatmaps) / len(heatmaps)

        return {
            'heatmap': heatmap,
        }


class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args)
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True):
        data = batch['image'].to(self.device)
        for key, value in batch.items():
            if value is not None:
                if hasattr(value, 'to'):
                    batch[key] = value.to(self.device)
        data = data.float()
        pred = self.model(data, training=training)

        if self.training:
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred


class SequenceRecognitionModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        nn.Module.__init__(self)

        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return '/' + args['backbone'] + '/' + args['decoder']

    def forward(self, batch, training=True):
        images = batch['image'].to(self.device)
        if self.training:
            labels = batch['label'].to(self.device)
            lengths = batch['length'].to(self.device).type(torch.long)
            loss, pred = self.model(
                images, targets=labels, lengths=lengths, train=True)
            return loss, pred
        else:
            return self.model(images, train=False)


class SegRecognitionModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        nn.Module.__init__(self)

        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return '/' + args['backbone'] + '/' + args['decoder']

    def forward(self, batch, training=True):
        images = batch['image'].to(self.device)
        if self.training:
            mask = batch['mask'].to(self.device)
            classify = batch['classify'].to(self.device).type(torch.long)
            return self.model(
                images, mask=mask, classify=classify, train=True)
        else:
            return self.model(images, train=False)


class IntegralRegressionRecognitionModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        nn.Module.__init__(self)

        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return '/' + args['backbone'] + '/' + args['decoder']

    def forward(self, batch, training=True):
        args = dict()
        for key in batch.keys():
            args[key] = batch[key].to(self.device)
        images = args.pop('image')
        return self.model(images, **args)


class GridSamplingModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        nn.Module.__init__(self)

        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return '/' + args['backbone'] + '/' + args['decoder']

    def forward(self, batch, training=True):
        args = dict()
        for key in batch.keys():
            args[key] = batch[key].to(self.device)
        images = args.pop('image')
        return self.model(images, **args)


class MaskRCNNTestModel(object):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        nn.Module.__init__(self)

        res = {}
        for respath in glob.glob(args['output-dir'] + '/res_*.txt'):
            name = os.path.basename(respath)[4:-4]
            polys = []
            with open(respath) as f:
                for line in f:
                    poly = np.array(list(map(int, line.strip().split(',')))).reshape(4, 2).tolist()
                    polys.append(poly)
            res[name] = polys
        self.res = res

    def eval(self):
        pass

    def forward(self, batch, training=False):
        assert (not training)

        image, label, meta = batch

        meta = [pickle.loads(value) for value in meta]

        results = []
        for image_meta in meta:
            name = image_meta['filename'].split('.')[0]
            pred_polys = self.res[name]
            results.append(pred_polys)

        return results
