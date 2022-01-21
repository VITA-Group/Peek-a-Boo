import argparse
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models

from models.seed_conv import SeedConv2d
from models.masked_psg_seed_conv import PredictiveSeedConv2d
from generator import masked_parameters
from prune import prune_loop


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help=('model architecture: '
                          + ' | '.join(model_names)
                          + ' (default: resnet18)'))
parser.add_argument('--init-method', default='standard', type=str,
                    help='initialization method for conv weights')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default='SGD', type=str,
                    help='choose among [`SGD`, `BOP`, `Counter`]')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--savedir', default='results', type=str,
                    help='root dir to save exp checkpoints and logs')
parser.add_argument('--pruned-save-name',
                    default='results/pruned_model.pth', type=str,
                    help='where to save the pruned models')
parser.add_argument('--exp-name', default='SeedNet', type=str,
                    help='path to location to save logs and checkpoints')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Options for models w/ latent weights
parser.add_argument('--hidden-act', type=str, default='standard',
                    help=('choose among '
                          '[`pruning`, `flipping`, `ternery`, `none`]'))
parser.add_argument('--scaling-input', action='store_true',
                    help='whether scale the input in SeedNet models')

# BOP options
parser.add_argument('--ar', type=float,
                    help='list of layer-wise inital adaptivity rates in BOP')
parser.add_argument('--tau', type=float,
                    help='list of layer-wise thresholds in BOP')
parser.add_argument('--ar-decay-freq', type=int, default=100,
                    help='freqency to decay the ar hyperparameter in BOP')
parser.add_argument('--ar-decay-ratio', type=float, default=0.1,
                    help='decay ratio when decay ar')

# PSG options
parser.add_argument('--psg-no-backward', action='store_true',
                    help='Do predictive gradient calculation in backward')
parser.add_argument('--msb-bits', type=int, default=4,
                    help='MSB bits for the input')
parser.add_argument('--msb-bits-weight', type=int, default=4,
                    help='MSB bits for the weight')
parser.add_argument('--msb-bits-grad', type=int, default=8,
                    help='MSB bits for the grad')
parser.add_argument('--psg-threshold', type=float, default=0.0,
                    help='Threshold used in PSG')
parser.add_argument('--psg-sparsify', action='store_true',
                    help='Sparsify by ignoring small gradients')
parser.add_argument('--psg-no-take-sign', action='store_true',
                    help='Do not take sign for PSG')

# Pruning options
parser.add_argument('--pruner', type=str, default=None,
                    choices=['Mag', 'SNIP', 'GraSP', 'SynFlow'],
                    help='pruning strategy')
parser.add_argument('--prune-epoch', type=int, default=0,
                    help='epoch number to finish sparsifying by')
parser.add_argument('--prune-ratio', type=float, default=1.0,
                    help='fraction of non-zero parameters after pruning')
parser.add_argument('--prune-iters', type=int, default=1,
                    help=('number of iterations for scoring '
                          '(should be 1 for Mag, SNIP, and GraSP)'))
parser.add_argument('--prune-batch-size', type=int, default=256,
                    help='size of sample mini-batch for pruning methods')
parser.add_argument('--prune-schedule', type=str, default='exponential',
                    choices=['linear', 'exponential'],
                    help='scheduling method for iterative pruning (SynFlow)')
parser.add_argument('--prune-scope', type=str, default='global',
                    choices=['global', 'local'],
                    help='masking scope')
parser.add_argument('--prune-shots', type=int, default=1,
                    help='number of shots for pruning')
parser.add_argument('--prune-verbose', action='store_true',
                    help='print additional information during pruning')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('psg'):
        model = models.__dict__[args.arch](
            init_method=args.init_method,
            predictive_backward = not args.psg_no_backward,
            msb_bits = args.msb_bits,
            msb_bits_weight = args.msb_bits_weight,
            msb_bits_grad = args.msb_bits_grad,
            threshold = args.psg_threshold,
            sparsify = args.psg_sparsify,
            sign = not args.psg_no_take_sign
        ).cpu()
        temp_arch = args.arch[9:] if 'seed' in args.arch else args.arch[4:]
        model_for_pruning = models.__dict__[temp_arch](init_method=args.init_method).cpu()
    else:
        model = models.__dict__[args.arch](init_method=args.init_method).cpu()
        model_for_pruning = None

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    sample_batch_indices = torch.randperm(len(train_dataset))[:100]
    sample_batch = torch.utils.data.Subset(train_dataset, sample_batch_indices)
    pruneloader = torch.utils.data.DataLoader(
        sample_batch, args.prune_batch_size, shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss().cpu()

    # Get all Convolutional layers in a model w/ latent weights
    seed_convs = list(
        filter(lambda m: isinstance(m, (SeedConv2d, PredictiveSeedConv2d,)),
               model.modules())
    )
    # Enable gradients for pruning for latent weight based methods
    for seed_conv in seed_convs:
        seed_conv.enable_weight_grad()
    num_classes = 1000
    """
    NOTE: We need to disable distributed training and perform pruning on CPU to
            avoid failures of the pruning methods on ImageNet models.
    """
    if args.arch.lower().startswith('psg'):
        # If the target network uses PSG, prune its non-PSG equivalence and
        #   load the pruned weights to the original PSG model.
        model_for_pruning.load_state_dict(model.state_dict(), strict=False)
        prune_loop(model_for_pruning, criterion, args.pruner,
                   pruneloader, num_classes, 'cpu', args.prune_ratio,
                   args.prune_schedule, args.prune_scope, args.prune_iters,
                   prune_bias=False, prune_batchnorm=False, prune_residual=False,
                   prune_verbose=args.prune_verbose)
        model.load_state_dict(model_for_pruning.state_dict(), strict=False)
    else:
        prune_loop(model, criterion, args.pruner,
                   pruneloader, num_classes, 'cpu', args.prune_ratio,
                   args.prune_schedule, args.prune_scope, args.prune_iters,
                   prune_bias=False, prune_batchnorm=False, prune_residual=False,
                   prune_verbose=args.prune_verbose)

    # Disable gradients when resuming training for SeedNet
    for seed_conv in seed_convs:
        seed_conv.disable_weight_grad()

    dirname = os.path.dirname(args.pruned_save_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    savedict = {
        'epoch': 0,
        'best_acc1': 0,
        'state_dict': model.state_dict(),
        'optimizer': None
    }
    torch.save(savedict, args.pruned_save_name)

if __name__ == "__main__":
    main()

