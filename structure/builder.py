from collections import OrderedDict

import torch

import structure.model
from structure.ensemble_model import EnsembleModel
from concern.config import Configurable, State


class Builder(Configurable):
    model = State()
    model_args = State()

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        if 'backbone' in cmd:
            self.model_args['backbone'] = cmd['backbone']

    @property
    def model_name(self):
        return self.model + '-' + getattr(structure.model, self.model).model_name(self.model_args)

    def build(self, device, distributed=False, local_rank: int = 0):
        Model = getattr(structure.model,self.model)
        model = Model(self.model_args, device,
                      distributed=distributed, local_rank=local_rank)
        return model


class EnsembleBuilder(Configurable):
    '''Ensemble multiple models into one model
    Input:
        builders: A dict which consists of several builders.
    Example:
        >>> builder:
                class: EnsembleBuilder
                builders:
                    ctc:
                        model: CTCModel
                    atten:
                        model: AttentionDecoderModel
    '''
    builders = State(default={})

    def __init__(self, cmd={}, **kwargs):
        resume_paths = dict()
        for key, value_dict in kwargs['builders'].items():
            resume_paths[key] = value_dict.pop('resume')
        self.resume_paths = resume_paths
        self.load_all(**kwargs)

    @property
    def model_name(self):
        return 'ensembled-model'

    def build(self, device, *args, **kwargs):
        models = OrderedDict()
        for key, builder in self.builders.items():
            models[key] = builder.build(device=device, *args, **kwargs)
            models[key].load_state_dict(torch.load(
                self.resume_paths[key], map_location=device),
                strict=True)
        return EnsembleModel(models)
