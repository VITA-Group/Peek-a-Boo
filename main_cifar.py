'''Train CIFAR10 with PyTorch.'''
from os import write
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import shutil
import argparse

import models
from models.seed_conv import SeedConv2d
from models.masked_psg_seed_conv import PredictiveSeedConv2d
from prune import prune_loop

from logger import set_logging_config
from bop import Bop

model_names = sorted(
    name for name in models.__dict__ 
    if not name.startswith("__") and callable(models.__dict__[name])
)


""" Create argument parser """
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--arch', default="ResNet18", type=str,
                    help='network architecture')
parser.add_argument('--init-method', default='standard', type=str,
                    help='initialization method for conv weights')
parser.add_argument('--dataset', default="cifar10", type=str,
                    help='Dataset choice')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum for SGD')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    help='weight decay for SGD')
parser.add_argument('--bs', default=128, type=int,
                    help='batch size')
parser.add_argument('--epochs', default=350, type=int,
                    help='max nubmer of epochs')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--load-pretrained', type=str, default=None,
                    help='load a pretrained network')
parser.add_argument('--optimizer', default='SGD', type=str,
                    help='choose among [`SGD`, `BOP`, `Counter`]')
parser.add_argument('--milestones', default='80,120', type=str,
                    help='Milestones for learning decay')
parser.add_argument('--savedir', default='results', type=str,
                    help='root dir to save exp checkpoints and logs')
parser.add_argument('--tsbdir', default='runs', type=str,
                    help='root dir to save tensorboard logs')
parser.add_argument('--exp-name', default='Peek-a-Boo', type=str,
                    help='path to location to save logs and checkpoints')

# Options for models w/ latent weights
parser.add_argument('--hidden-act', type=str, default='standard',
                    help=('choose among '
                          '[`pruning`, `flipping`, `ternery`, `none`]'))
parser.add_argument('--scaling-input', action='store_true',
                    help=('whether scale the input '
                          'in models with latent weights'))

# BOP options
parser.add_argument('--ar', type=float, default=1e-3,
                    help='inital adaptivity rate in BOP')
parser.add_argument('--tau', type=float, default=1e-6,
                    help='threshold in BOP')
parser.add_argument('--ar-decay-freq', type=int, default=100,
                    help='freqency to decay the ar hyperparameter in BOP')
parser.add_argument('--ar-decay-ratio', type=float, default=0.1,
                    help='decay ratio when decay ar')
parser.add_argument('--bop-threshold-scale', type=float, default=0.1,
                    help='threshold scale for large weights in ScaledBop')
parser.add_argument('--scaled-mag-quantile', type=float, default=0.5,
                    help='quatile of magnitudes in ScaledBop')

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
                    choices=['Mag', 'SNIP', 'GraSP', 'SynFlow', 'GradFlow'],
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

parser.add_argument('--seed', default=0, type=int,
                    help='random seed for reproducable initialization')
parser.add_argument('--use-cuda', type=str, default=None,
                    help='gpu id for multi-gpu sweep')

""" Experiment setup """
args = parser.parse_args()

global_step = 0

args.savedir = os.path.join(args.savedir, args.exp_name)
args.tsbdir = os.path.join(args.tsbdir, args.exp_name)
if not os.path.isdir(args.savedir):
    os.makedirs(args.savedir)
if not os.path.isdir(args.tsbdir):
    os.makedirs(args.tsbdir)

logger = set_logging_config(args.savedir)

device = (f'cuda:{args.use_cuda}' if args.use_cuda and torch.cuda.is_available()
          else 'cpu')
best_acc = 0  # best test accuracy
best_acc_epoch = 0  # the epoch where the best test accuracy happens
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
torch.manual_seed(args.seed)

""" Create dataloader """
logger.info('==> Preparing data..')

if args.dataset.lower() == 'cifar10':
    dataset_mean = (0.4914, 0.4822, 0.4465)
    dataset_std  = (0.2023, 0.1994, 0.2010)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 10
elif args.dataset.lower() == 'cifar100':
    dataset_mean = (0.507, 0.487, 0.441)
    dataset_std  = (0.267, 0.256, 0.276)
    classes = (None,)
    num_classes = 100

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std),
])

if args.dataset.lower() == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform_test)
elif args.dataset.lower() == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, 
        download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR100(
        root='./data', train=False,
        download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.bs, shuffle=True, num_workers=8)
testloader  = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8)

# Dataloader for data-dependent pruning methods, such as SNIP and GraSP
sample_batch_indices = torch.randperm(len(trainset))[:100]
sample_batch = torch.utils.data.Subset(trainset, sample_batch_indices)
pruneloader = torch.utils.data.DataLoader(
    sample_batch, args.prune_batch_size, shuffle=True, num_workers=8)


""" Build the network model """
logger.info('==> Building model..')

if args.arch.startswith('VGG'):
    net = models.__dict__['VGG'](
        args.arch,
        num_classes=num_classes,
        init_method=args.init_method
    )
elif "Shuffle" in args.arch:
    net = models.__dict__[args.arch](1)
    pass
elif args.arch.startswith('SeedVGG'):
    net = models.__dict__['SeedVGG'](
        args.arch,
        args.sign_grouped_dim,
        args.init_method,
        args.hidden_act,
        args.scaling_input,
        num_classes=num_classes
    )
elif args.arch.startswith('SeedResNet'):
    net = models.__dict__[args.arch](
        args.sign_grouped_dim,
        args.init_method,
        args.hidden_act,
        args.scaling_input,
        num_classes=num_classes
    )
elif args.arch.startswith('PsgVGG'):
    net = models.__dict__['PsgVGG'](
        args.arch,
        num_classes = num_classes,
        init_method = args.init_method,
        predictive_backward = not args.psg_no_backward,
        msb_bits = args.msb_bits,
        msb_bits_weight = args.msb_bits_weight,
        msb_bits_grad = args.msb_bits_grad,
        threshold = -args.psg_threshold,
        sparsify = args.psg_sparsify,
        sign = not args.psg_no_take_sign
    )
elif args.arch.startswith('PsgSeedVGG'):
    net = models.__dict__['PsgSeedVGG'](
        args.arch,
        num_classes = num_classes,
        init_method = args.init_method,
        predictive_backward = not args.psg_no_backward,
        msb_bits = args.msb_bits,
        msb_bits_weight = args.msb_bits_weight,
        msb_bits_grad = args.msb_bits_grad,
        threshold = -args.psg_threshold,
        sparsify = args.psg_sparsify,
        sign = not args.psg_no_take_sign
    )
elif "Psg" in args.arch:
    net = models.__dict__[args.arch](
        num_classes = num_classes,
        init_method = args.init_method,
        predictive_backward = not args.psg_no_backward,
        msb_bits = args.msb_bits,
        msb_bits_weight = args.msb_bits_weight,
        msb_bits_grad = args.msb_bits_grad,
        threshold = -args.psg_threshold,
        sparsify = args.psg_sparsify,
        sign = not args.psg_no_take_sign
    )
else:
    net = models.__dict__[args.arch](num_classes=num_classes,
                                     init_method=args.init_method)

net = net.to(device)
# Create a non-PSG equivalence for pruning if the target network is a PSG model
if args.arch.startswith('Psg'):
    if 'VGG' in args.arch:
        net_for_pruning = models.__dict__['VGG'](
            args.arch[-5:],
            num_classes=num_classes,
            init_method=args.init_method
        )
    else:
        temp_arch = args.arch.strip('Psg').strip('Seed')
        net_for_pruning = models.__dict__[temp_arch](
            num_classes=num_classes, init_method=args.init_method
        )
    net_for_pruning.to(device)
else:
    net_for_pruning = None

if os.path.exists(os.path.join(args.savedir, 'init.pth')):
    logger.info('==> An initialization checkpoint exists. '
                'Resuming from the initialization..')
    net_to_load = net.module if hasattr(net, 'module') else net
    checkpoint = torch.load(os.path.join(args.savedir, 'init.pth'), 
                            map_location='cpu')
    net_to_load.load_state_dict(checkpoint['net'])
    best_acc = 0.0
    start_epoch = 0
else:
    if args.load_pretrained:
        assert os.path.exists(args.load_pretrained)
        net_to_load = net.module if hasattr(net, 'module') else net
        checkpoint = torch.load(args.load_pretrained, map_location='cpu')
        if isinstance(checkpoint, dict) and 'net' in checkpoint.keys():
            net_to_load.load_state_dict(checkpoint['net'])
        else:
            net_to_load.load_state_dict(checkpoint)
    net_to_save = net.module if hasattr(net, 'module') else net
    logger.info('Saving initialization..')
    state = {'net': net_to_save.state_dict(), 'acc': 0.0, 'epoch': 0}
    ckpt_path = os.path.join(args.savedir, 'init.pth')
    torch.save(state, ckpt_path)

if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    net_to_load = net.module if hasattr(net, 'module') else net
    checkpoint = torch.load('./checkpoint/ckpt.pth', map_location='cpu')
    net_to_load.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

args.milestones = list(map(int, args.milestones.split(',')))

def build_optimizers():
    if args.optimizer == 'SGD':
        parameters = [p for p in net.parameters() if p.requires_grad]
        optimizers = (optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay),)
        schedulers = (optim.lr_scheduler.MultiStepLR(optimizers[0],
                                                     milestones=args.milestones,
                                                     gamma=0.1),)
    elif args.optimizer == 'BOP':
        bop_params, non_bop_params = net.get_bop_params(), net.get_non_bop_params()
        bop_param_masks = net.get_bop_param_masks()
        bop_optimizer = Bop(bop_params, None, bop_param_masks,
                            ar=args.ar, threshold=args.tau, device=device)
        if len(non_bop_params) > 0:
            non_bop_optimizer = optim.SGD(non_bop_params, lr=args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay)
            optimizers = (bop_optimizer, non_bop_optimizer,)
            schedulers = (optim.lr_scheduler.MultiStepLR(
                non_bop_optimizer, milestones=args.milestones, gamma=0.1),)
        else:
            optimizers = (bop_optimizer,)
            schedulers = ()
    else:
        raise NotImplementedError(f'Optimizer {args.optimizer} '
                                  'not implemented yet')
    return optimizers, schedulers

optimizers, schedulers = build_optimizers()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    if net_for_pruning is not None:
        net_for_pruning = torch.nn.DataParallel(net_for_pruning)
    cudnn.benchmark = True

# Training
def train(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global global_step
    with tqdm(trainloader, dynamic_ncols=True) as t:
        for batch_idx, (inputs, targets) in enumerate(t):

            inputs, targets = inputs.to(device), targets.to(device)
            for optimizer in optimizers:
                optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            t.set_postfix(loss=train_loss/(batch_idx+1),
                          acc=100.*correct/total,
                          correct=correct, total=total)


def test(epoch):
    global best_acc, best_acc_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(testloader, dynamic_ncols=True) as t:
            # for batch_idx, (inputs, targets) in enumerate(testloader):
            for batch_idx, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                t.set_postfix(loss=test_loss/(batch_idx+1),
                              acc=100.*correct/total,
                              correct=correct, total=total)

            acc = 100.*correct/total
            logger.info('Epoch: %d\t Test Accuracy: %2.2f' % (epoch, acc))

    # Save checkpoint.
    acc = 100.*correct/total
    net_to_save = net.module if hasattr(net, 'module') else net
    logger.info('Saving..')
    state = {'net': net_to_save.state_dict(), 'acc': acc, 'epoch': epoch}
    ckpt_path = os.path.join(args.savedir, 'ckpt.pth')
    torch.save(state, ckpt_path)
    if acc > best_acc:
        shutil.copy(ckpt_path, os.path.join(args.savedir, 'ckpt_best.pth'))
        best_acc = acc
        best_acc_epoch = epoch

# Get all Convolutional layers in a model w/ latent weights
seed_convs = list(
    filter(lambda m: isinstance(m, (SeedConv2d, PredictiveSeedConv2d,)), 
           net.modules())
)
cur_shot = 0
prune_interval = int(args.prune_epoch / args.prune_shots)
for epoch in range(start_epoch, args.epochs):
    if (args.pruner and
            epoch == (cur_shot + 1) * prune_interval and
            cur_shot < args.prune_shots):
        target_sparsity = (
            1 - (1 - args.prune_ratio) * (cur_shot + 1) / args.prune_shots
        )

        # Enable gradients for pruning for latent weight based methods
        for seed_conv in seed_convs:
            seed_conv.enable_weight_grad()

        if args.arch.lower().startswith('psg'):
            # If the target network uses PSG, prune its non-PSG equivalence and
            #   load the pruned weights to the original PSG model.
            net_for_pruning.load_state_dict(net.state_dict(), strict=False)
            prune_loop(net_for_pruning, criterion, args.pruner,
                       pruneloader, num_classes, device, target_sparsity,
                       args.prune_schedule, args.prune_scope, args.prune_iters,
                       prune_bias=False, prune_batchnorm=False, prune_residual=False,
                       prune_verbose=args.prune_verbose)
            net.load_state_dict(net_for_pruning.state_dict(), strict=False)
        else:
            prune_loop(net, criterion, args.pruner,
                       pruneloader, num_classes, device, target_sparsity,
                       args.prune_schedule, args.prune_scope, args.prune_iters,
                       prune_bias=False, prune_batchnorm=False, prune_residual=False,
                       prune_verbose=args.prune_verbose)

        # Disable gradients when resuming training for latent weight based methods
        for seed_conv in seed_convs:
            seed_conv.disable_weight_grad()

        cur_shot += 1

    train(epoch)
    test(epoch)

    # schedule learning rates
    for scheduler in schedulers:
        scheduler.step()

    # schedule the adaptivity rate in BOP
    if 'BOP' in args.optimizer and (epoch + 1) % args.ar_decay_freq == 0:
        optimizers[0].decay_ar(args.ar_decay_ratio)

logger.info('best acc {} at epoch {}'.format(best_acc, best_acc_epoch))
