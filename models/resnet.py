# encoding: utf-8
# author: Gofinge

import torch.nn as nn
import torchvision

__all__ = ['resnet50', 'resnet101', 'resnet152']


def resnet50(num_classes, pretrained=False, **kwargs):
    model = torchvision.models.resnet50(pretrained=pretrained)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    model = torchvision.models.resnet101(pretrained=pretrained)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    model = torchvision.models.resnet152(pretrained=pretrained)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )
    return model


if __name__ == '__main__':
    # For Mac: run '/Applications/Python\ 3.7/Install\ Certificates.command' is necessary
    assert resnet50(num_classes=10, pretrained=True)
    print('success')
    assert resnet101(num_classes=10, pretrained=True)
    print('success')
    assert resnet152(num_classes=10, pretrained=True)
    print('success')