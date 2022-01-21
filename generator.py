from masked_layers import layers
from models.masked_psg_conv import PredictiveConv2d
from models.masked_psg_seed_conv import PredictiveSeedConv2d
from models.seed_conv import SeedConv2d

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf

def trainable(module):
    r"""Returns boolean whether a module is trainable.
    """
    return not isinstance(module, (layers.Identity1d, layers.Identity2d))

def prunable(module, batchnorm, residual):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (layers.Linear, layers.Conv2d, PredictiveConv2d, SeedConv2d, PredictiveSeedConv2d))
    if batchnorm:
        isprunable |= isinstance(module, (layers.BatchNorm1d, layers.BatchNorm2d))
    if residual:
        isprunable |= isinstance(module, (layers.Identity1d, layers.Identity2d))
    return isprunable

def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in filter(lambda p: trainable(p), model.modules()):
        for param in module.parameters(recurse=False):
            yield param

def masked_parameters(model, bias=False, batchnorm=False, residual=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            if param is not module.bias or bias is True:
                yield mask, param
