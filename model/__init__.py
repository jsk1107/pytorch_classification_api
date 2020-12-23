from efficientnet_pytorch import EfficientNet
from torchvision.models import *

def network(model, pretrained=True, num_classes=None):
    if model == 'resnet-18':
        model = resnet18(pretrained=pretrained, progress=True)
    elif model == 'resnet-34':
        model = resnet34(pretrained=pretrained, progress=True)
    elif model == 'resnet-50':
        model = resnet50(pretrained=pretrained, progress=True)
    elif model == 'resnet-101':
        model = resnet101(pretrained=pretrained, progress=True)
    elif model == 'resnet-152':
        model = resnet152(pretrained=pretrained, progress=True)
    elif model == 'resnext-50':
        model = resnext50_32x4d(pretrained=pretrained, progress=True)
    elif model == 'resnext-101':
        model = resnext101_32x8d(pretrained=pretrained, progress=True)
    elif model == 'vgg-19':
        model = vgg19(pretrained=pretrained, progress=True)
    elif model == 'inception-v3':
        model = inception_v3(pretrained=pretrained, progress=True)
    elif model == 'mobilenet-v2':
        model = mobilenet_v2(pretrained=pretrained, progress=True)
    elif model == 'mobilenet-v2':
        model = mobilenet_v2(pretrained=pretrained, progress=True)
    elif model == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    elif model == 'efficientnet-b1':
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
    elif model == 'efficientnet-b2':
        model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
    elif model == 'efficientnet-b3':
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
    elif model == 'efficientnet-b4':
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    elif model == 'efficientnet-b5':
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
    elif model == 'efficientnet-b6':
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
    elif model == 'efficientnet-b7':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
    else:
        raise ImportError(f'=========> It is not found the {model}')

    return model