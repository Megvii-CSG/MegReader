import torch
import cv2
import numpy as np

from concern.config import Configurable, State


class TrivalVisualizer(Configurable):
    def __init__(self, **kwargs):
        pass

    def visualize(self, batch, output):
        return {}
