# encoding: utf-8
# author: Gofinge

import torch.nn as nn
import torchvision

__all__ = ['densenet121', 'densenet169', 'densenet201']


def densenet121(num_classes, pretrained=False, **kwargs):
    model = torchvision.models.densenet121(pretrained=pretrained, **kwargs)

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.Dropout(p=0.9),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )
    return model


def densenet169(num_classes, pretrained=False, **kwargs):
    model = torchvision.models.densenet169(pretrained=pretrained, **kwargs)

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.Dropout(p=0.9),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )
    return model


def densenet201(num_classes, pretrained=False, **kwargs):
    model = torchvision.models.densenet201(pretrained=pretrained, **kwargs)

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.Dropout(p=0.9),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()

    )
    return model


if __name__ == '__main__':
    # For Mac: run '/Applications/Python\ 3.7/Install\ Certificates.command' is necessary
    assert densenet121(num_classes=10, pretrained=True)
    print('success')
    assert densenet169(num_classes=10, pretrained=True)
    print('success')
    assert densenet201(num_classes=10, pretrained=True)
    print('success')
