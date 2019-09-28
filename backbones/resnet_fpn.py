from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .fpn_top_down import FPNTopDown
from .feature_pyramid import FeaturePyramid


def Resnet18FPN(resnet_pretrained=True):
    bottom_up = resnet18(pretrained=resnet_pretrained)
    top_down = FPNTopDown([512, 256, 128, 64], 256)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid


def Resnet34FPN(resnet_pretrained=True):
    bottom_up = resnet50(pretrained=resnet_pretrained)
    top_down = FPNTopDown([2048, 1024, 512, 256], 256)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid


def Resnet50FPN(resnet_pretrained=True):
    bottom_up = resnet50(pretrained=resnet_pretrained)
    top_down = FPNTopDown([2048, 1024, 512, 256], 256)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid


def Resnet101FPN(resnet_pretrained=True):
    bottom_up = resnet101(pretrained=resnet_pretrained)
    top_down = FPNTopDown([2048, 1024, 512, 256], 256)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid


def Resnet152FPN(resnet_pretrained=True):
    bottom_up = resnet152(pretrained=resnet_pretrained)
    top_down = FPNTopDown([2048, 1024, 512, 256], 256)
    feature_pyramid = FeaturePyramid(bottom_up, top_down)
    return feature_pyramid
