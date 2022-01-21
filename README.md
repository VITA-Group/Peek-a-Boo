# Peek-a-Boo: What (More) is Disguised in a Randomly Weighted Neural Network, and How to Find It Efficiently

This repository is the official implementation for the following paper
Analytic-LISTA networks proposed in the following paper:

"Peek-a-Boo: What (More) is Disguised in a Randomly Weighted Neural Network, and How to Find It Efficiently" by [Xiaohan Chen](http://www.xiaohanchen.com/), Jason Zhang and Zhangyang Wang from the [VITA Research Group](https://vita-group.github.io/).

The code implements the Peek-a-Boo (*PaB*) algorithm for various convolutional networks  and is tested in Linux environment with Python: 3.7.2, PyTorch 1.7.0+.

## Getting Started 

### Dependency

```
pip install tqdm
```

### Prerequisites
- Python 3.7+
- PyTorch 1.7.0+
- tqdm

### Data Preparation

To run ImageNet experiments, download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively as shown below. A useful script for automatic extraction can be found [here](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## How to Run Experiments

### CIFAR-10/100 Experiments

To apply PaB w/ PSG to a ResNet-18 network on CIFAR-10/100 datasets, use the following command:

```bash
python main.py --use-cuda 0 \
    --arch PsgResNet18 --init-method kaiming_normal \
    --optimizer BOP --ar 1e-3 --tau 1e-6 \
    --ar-decay-freq 45 --ar-decay-ratio 0.15 --epochs 180 \
    --pruner SynFlow --prune-epoch 0 \
    --prune-ratio 3e-1 --prune-iters 100 \
    --msb-bits 8 --msb-bits-weight 8 --msb-bits-grad 16 \
    --psg-threshold 1e-7 --psg-no-take-sign --psg-sparsify \
    --exp-name cifar10_resnet18_pab-psg
```

To break down the above complex command, PaB includes two stages (pruning and Bop training) and consists of three components (a pruner, a Bop optimizer and a PSG module).

[**Pruning module**] The pruning module is controlled by the following arguments:

* `--pruner` - A string that indicates which pruning method to be used. Valid choices are `['Mag', 'SNIP', 'GraSP', 'SynFlow']`.
* `--prune-epoch` - An integer, the epoch index of when (the last) pruning is performed.
* `--prune-ratio` - A float, the ratio of non-zero parameters remained after (the last) pruning
* `--prune-iters` - An integeer, the number of pruning iterations in one run of pruning. Check the [SynFlow paper](https://arxiv.org/abs/2006.05467) for what this means.

[**Bop optimizer**] Bop has several hyperparameters that are essential to its successful optimizaiton as shown below. More details can be found in the original [Bop paper](https://arxiv.org/abs/1906.02107).

* ``--optimizer`` - A string that specifies the Bop optimizer. You can pass `'SGD'` to this argument for a standard training of SGD. Check here.
* ``--ar`` - A float, corresponding to the adativity rate for the calculation of gradient moving average.
* ``--tau`` - A float, corresponding to the threshold that decides if a binary weight needs to be flipped.
* ``--ar-decay-freq`` - An integer, interval in epochs between decays of the adaptivity ratio.
* ``--ar-decay-ratio`` - A float, the decay ratio of the adaptivity ratio decaying.

[**PSG module**] PSG stands for *Predictive Sign Gradient*, which was originally proposed in the [E2-Train paper](https://arxiv.org/abs/1910.13349). PSG uses low-precision computation during backward passes to save computational cost. It is controlled by several arguments.

* ``--msb-bits, --msb-bits-weight, --msb-bits-grad`` - Three floats, the bit-width for the inputs, weights and output errors during back-propagation.
* ``--psg-threshold`` - A float, the threshold that filters out coarse gradients with small magnitudes to reduce gradient variance.
* ``--psg-no-take-sign`` - A boolean that indicates to bypass the "taking-the-sign" step in the original PSG method.
* ``--psg-sparsify`` - A boolean. The filtered small gradients are set to zero when it is true.

### ImageNet Experiments

For PaB experiments on ImageNet, we run the pruning and Bop training in a two-stage manner, implemented in `main_imagenet_prune.py` and `main_imagenet_train.py`, respectively.

To prune a ResNet-50 network at its initialization, we first run the following command to perform SynFlow, which follows a similar manner for the arguments as in CIFAR experiments:

```bash
export prune_ratio=0.5  # 50% remaining parameters.

# Run SynFlow pruning
python main_imagenet_prune.py \
    --arch resnet50 --init-method kaiming_normal \
    --pruner SynFlow --prune-epoch 0 \
    --prune-ratio $prune_ratio --prune-iters 100 \
    --pruned-save-name /path/to/the/pruning/output/file \
    --seed 0 --workers 32 /path/to/the/imagenet/dataset
```

We then train the pruned model using Bop with PSG on one node with multi-GPUs.

```bash
# Bop hyperparameters
export bop_ar=1e-3
export bop_tau=1e-6
export psg_threshold="-5e-7"

python main_imagenet_train.py \
    --arch psg_resnet50 --init-method kaiming_normal \
    --optimizer BOP --ar $bop_ar --tau $bop_tau \
    --ar-decay-freq 30 --ar-decay-ratio 0.15 --epochs 100 \
    --msb-bits 8 --msb-bits-weight 8 --msb-bits-grad 16 \
    --psg-sparsify --psg-threshold " ${psg_threshold}" --psg-no-take-sign \
    --savedir /path/to/the/output/dir \
    --resume /path/to/the/pruning/output/file \
    --exp-name 'imagenet_resnet50_pab-psg' \
    --dist-url 'tcp://127.0.0.1:2333' \
    --dist-backend 'nccl' --multiprocessing-distributed \
    --world-size 1 --rank 0 \
    --seed 0 --workers 32 /path/to/the/imagenet/dataset 
```


## Cite this work

If you find this work or our code implementation helpful for your own resarch or work, please cite our paper.

```
@inproceedings{
chen2022peek,
title={Peek-a-Boo: What (More) is Disguised in a Randomly Weighted Neural Network, and How to Find It Efficiently},
author={Xiaohan Chen and Jason Zhang and Zhangyang Wang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=moHCzz6D5H3},
}
```
