from torchvision import models


def load_resnet(name: str, pretrained: bool = True):
    """Load a ResNet backbone with torchvision's current weights API."""
    model_fn = getattr(models, name)
    if not pretrained:
        return model_fn(weights=None)

    weights_cls_name = f"ResNet{name.replace('resnet', '')}_Weights"
    weights_cls = getattr(models, weights_cls_name, None)
    if weights_cls is not None:
        return model_fn(weights=weights_cls.DEFAULT)

    return model_fn(pretrained=True)
