import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models

import logging
from logger import set_logging_config
import models
from bop import Bop

from models.seed_conv import SeedConv2d
from models.masked_psg_seed_conv import PredictiveSeedConv2d
import pruners
from generator import masked_parameters
from prune import prune_loop


print = print

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
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
# SeedNet options
parser.add_argument('--sign-grouped-dim', default="", type=str,
                    help='dimensions that will be grouped for sign parameters')
parser.add_argument('--init-method', default='standard', type=str,
                    help='initialization method for conv weights')
parser.add_argument('--hidden-act', type=str, default='standard',
                    help='choose among [`pruning`, `flipping`, `ternery`, `none`]')
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
parser.add_argument('--pruner', type=str, default=None, choices=['Mag', 'SNIP', 'GraSP', 'SynFlow'],
                    help='pruning strategy')
parser.add_argument('--prune-epoch', type=int, default=0,
                    help='epoch number to finish sparsifying by')
parser.add_argument('--prune-ratio', type=float, default=1.0,
                    help='fraction of non-zero parameters after pruning')
parser.add_argument('--prune-iters', type=int, default=1,
                    help='number of iterations for scoring (should be 1 for Mag, SNIP, and GraSP)')
parser.add_argument('--prune-batch-size', type=int, default=256,
                    help='size of sample mini-batch for pruning methods')
parser.add_argument('--prune-schedule', type=str, default='exponential', choices=['linear', 'exponential'],
                    help='scheduling method for iterative pruning (SynFlow)')
parser.add_argument('--prune-scope', type=str, default='global', choices=['global', 'local'],
                    help='masking scope')
parser.add_argument('--prune-shots', type=int, default=1,
                    help='number of shots for pruning')

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

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    args.savedir = os.path.join(args.savedir, args.exp_name)
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    args.logger = set_logging_config(args.savedir)

    if args.gpu is not None:
        args.logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        args.logger.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        args.logger.info("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('seed_resnet'):
            pass
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
            )
            temp_arch = args.arch[9:] if 'seed' in args.arch else args.arch[4:]
            model_for_pruning = models.__dict__[temp_arch](init_method=args.init_method)
        else:
            model = models.__dict__[args.arch](init_method=args.init_method)
            model_for_pruning = None

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
            if model_for_pruning is not None:
                model_for_pruning.cuda(args.gpu)
                model_for_pruning = torch.nn.parallel.DistributedDataParallel(model_for_pruning, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
            if model_for_pruning is not None:
                model_for_pruning.cuda()
                model_for_pruning = torch.nn.parallel.DistributedDataParallel(model_for_pruning)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'SGD':
        parameters = [p for p in model_without_ddp.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        bop_optimizer = None
    elif args.optimizer == 'BOP':
        bop_params, non_bop_params = model_without_ddp.get_bop_params(), model_without_ddp.get_non_bop_params()
        bop_param_masks = model_without_ddp.get_bop_param_masks()
        bop_dict = [{'params': bop_params, 'adaptivity_rate': args.ar, 'threshold': args.tau}]
        # optimizer = optim.SGD(non_bop_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.SGD(non_bop_params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # bop_optimizer = Bop(bop_params, None, ar=args.ar, threshold=args.tau)
        bop_optimizer = Bop(bop_params, None, bop_param_masks, ar=args.ar, threshold=args.tau, device=args.gpu)
        # schedulers = (optim.lr_scheduler.MultiStepLR(non_bop_optimizer, milestones=[80, 120], gamma=0.1),)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            args.logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model_without_ddp.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            args.logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
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

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    sample_batch_indices = torch.randperm(len(train_dataset))[:100]
    sample_batch = torch.utils.data.Subset(train_dataset, sample_batch_indices)
    pruneloader = torch.utils.data.DataLoader(sample_batch, args.prune_batch_size, shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # Create pruner
    num_classes = 1000
    # if args.pruner:
    #     pruner = pruners.__dict__[args.pruner](masked_parameters(model, False, False, False), num_classes)

    seed_convs = list(filter(lambda m: isinstance(m, (SeedConv2d, PredictiveSeedConv2d,)), model.modules()))
    cur_shot = 0
    prune_interval = int(args.prune_epoch / args.prune_shots)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        if args.optimizer == 'BOP' and (epoch + 1) % args.ar_decay_freq == 0:
            bop_optimizer.decay_ar(args.ar_decay_ratio)

        # Enable gradients for pruning in SeedNet
        for seed_conv in seed_convs:
            seed_conv.enable_weight_grad()
        if args.pruner and epoch == (cur_shot + 1) * prune_interval and cur_shot < args.prune_shots:
            target_sparsity = 1 - (1 - args.prune_ratio) * (cur_shot + 1) / args.prune_shots
            if args.arch.lower().startswith('psg'):
                model_for_pruning.load_state_dict(model.state_dict(), strict=False)
                # pruner = pruners.__dict__[args.pruner](masked_parameters(model_for_pruning, False, False, False), num_classes)
                # prune_loop(model_for_pruning, criterion, pruner, pruneloader, num_classes, args.gpu, target_sparsity,
                #            args.prune_schedule, args.prune_scope, args.prune_iters)
                prune_loop(model_for_pruning, criterion, args.pruner,
                           pruneloader, num_classes, args.gpu, target_sparsity,
                           args.prune_schedule, args.prune_scope, args.prune_iters,
                           prune_bias=False, prune_batchnorm=False, prune_residual=False,
                           weight_flips=None, score_threshold=None)
                model.load_state_dict(model_for_pruning.state_dict(), strict=False)
            else:
                # prune_loop(model, criterion, pruner, pruneloader, num_classes, args.gpu, target_sparsity,
                #            args.prune_schedule, args.prune_scope, args.prune_iters)
                prune_loop(model, criterion, args.pruner,
                           pruneloader, num_classes, args.gpu, target_sparsity,
                           args.prune_schedule, args.prune_scope, args.prune_iters,
                           prune_bias=False, prune_batchnorm=False, prune_residual=False,
                           weight_flips=None, score_threshold=None)
            # Really copy the mask to the model
            # with torch.no_grad():
            #     pruned_masks = [m for m, _ in pruner.masked_parameters]
            #     model_masks  = [m for m, _ in masked_parameters(model, False, False, False)]
            #     for model_mask, pruned_mask in zip(model_masks, pruned_masks):
            #         model_mask.copy_(pruned_mask.data.detach().clone())
            # Disable gradients when resuming training for SeedNet
            for seed_conv in seed_convs:
                seed_conv.disable_weight_grad()
            cur_shot += 1

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, bop_optimizer=bop_optimizer)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
        if args.gpu == 0:
            args.logger.info('epoch {} \t Top-1 acc {} \t Top-5 acc {}'.format(epoch + 1, acc1, acc5))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        args.logger.info(f'Max accuracy: {best_acc1}')
        best_acc1_acc5 = acc5

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_without_ddp.state_dict(),
                'best_acc1': best_acc1,
                'acc5': best_acc1_acc5,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

    args.logger.info('best Top-1 acc {} \t corresponding Top-5 acc {}'.format(best_acc1, best_acc1_acc5))


def train(train_loader, model, criterion, optimizer, epoch, args, bop_optimizer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if bop_optimizer is not None:
            bop_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if bop_optimizer is not None:
            bop_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.gpu == 0 and i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.gpu == 0 and i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        top1.synchronize()
        top5.synchronize()
        # args.logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        """
        Warning: does not synchronize `val`
        """
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = float(t[0])
        self.count = int(t[1])
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

