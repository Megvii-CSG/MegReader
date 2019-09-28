#!python3
import argparse
import os

import torch
import yaml
from tqdm import tqdm

from trainer import Trainer
import ipdb
# tagged yaml objects
from experiment import Structure, TrainSettings, ValidationSettings, Experiment
from concern.log import Logger
from data.data_loader import DataLoader
from data.mnist import MNistDataset
from data.nori_dataset import NoriDataset
from training.checkpoint import Checkpoint
from training.learning_rate import (
    ConstantLearningRate, PriorityLearningRate, FileMonitorLearningRate
)
from training.model_saver import ModelSaver
from training.optimizer_scheduler import OptimizerScheduler
from concern.config import Configurable, Config


def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--start_iter', type=int,
                        help='Begin counting iterations starting from this value (should be used with resume)')
    parser.add_argument('--start_epoch', type=int,
                        help='Begin counting epoch starting from this value (should be used with resume)')
    parser.add_argument('--max_size', type=int, help='max length of label')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--verbose', action='store_true',
                        help='show verbose info')
    parser.add_argument('--no-verbose', action='store_true',
                        help='show verbose info')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')
    parser.add_argument('--speed', action='store_true', dest='test_speed',
                        help='Test speed only')
    parser.add_argument('--dest', type=str,
                        help='Specify which prediction will be used for decoding.')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='Run with debug mode, which hacks dataset num_samples to toy number')
    parser.add_argument('--no-debug', action='store_false',
                        dest='debug', help='Run without debug mode')
    parser.add_argument('-d', '--distributed', action='store_true',
                        dest='distributed', help='Use distributed training')
    parser.add_argument('--local_rank', dest='local_rank', default=0,
                        type=int, help='Use distributed training')
    parser.add_argument('-g', '--num_gpus', dest='num_gpus', default=4,
                        type=int, help='The number of accessible gpus')
    parser.set_defaults(debug=False, verbose=False)

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Eval(experiment, experiment_args, cmd=args, verbose=args['verbose']).eval(args['visualize'])


class Eval:
    def __init__(self, experiment, args, cmd=dict(), verbose=False):
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.data_loaders = experiment.evaluation.data_loaders
        self.args = cmd
        self.logger = experiment.logger
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = cmd.get(
            'resume', os.path.join(
                self.logger.save_dir(model_saver.dir_path),
                'final'))
        self.verbose = verbose

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            self.logger.warning("Checkpoint not found: " + path)
            return
        self.logger.info("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        self.logger.info("Resumed from " + path)

    def report_speed(self, model, batch, times=100):
        import time
        data = {k: v[0:1]for k, v in batch.items()}
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(times):
            model.forward(data)
        if cuda:
            torch.cuda.synchronize()
        time_cost = (time.time() - start) / times
        self.logger.info('Params: %s, Inference speed: %fms, FPS: %f' % (
            str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            time_cost * 1000, 1 / time_cost))

    def eval(self, visualize=False):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        all_matircs = {}
        model.eval()
        vis_images = dict()
        with torch.no_grad():
            for name, data_loader in self.data_loaders.items():
                raw_metrics = []
                if self.args.get('data', None) and not self.args['data'] == name:
                    continue
                for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                    if self.args['test_speed']:
                        self.report_speed(model, batch)
                        continue

                    pred = model.forward(batch, training=False)
                    output = self.structure.representer.represent(batch, pred)
                    raw_metric, interested = self.structure.measurer.validate_measure(batch, output)
                    raw_metrics.append(raw_metric)

                    if visualize and self.structure.visualizer:
                        vis_image = self.structure.visualizer.visualize(batch, output, interested)
                        self.logger.save_image_dict(vis_image)
                        vis_images.update(vis_image)
                metrics = self.structure.measurer.gather_measure(raw_metrics, self.logger)
                for key, metric in metrics.items():
                    self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))
                    all_matircs[name + '/' + key] = metric
        for key, metric in all_matircs.items():
            self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))


if __name__ == '__main__':
    main()
