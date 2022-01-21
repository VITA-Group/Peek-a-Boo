import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from typing import Optional, Callable


class Bop(Optimizer):
    """Bop optimizer based on https://arxiv.org/abs/1906.02107.
    
    Attributes:
        binary_params:
            Binary weights that will be optimized by Bop.
        bn_params:
            Non-binary weights that will be optimized by SGD-type optimizers
            such as SGD and Adam.
            NOTE: This parameter is not used yet. The optimization of non-binary
            weights are doen by an external optimizer.
        prune_masks:
            Intended to store the pruning mask of the binary parameters.
            NOTE: Not used in this implementation.
        args:
            Other arguments for the Bop optimizer, including:
            - ar: Adaptivity rate for calculating the moving-averaged gradient.
            - tau: Threshold for the flipping decision.
        
    Returns:
        An instance of a Bop optimizer.
    """
    def __init__(
        self,
        binary_params,
        bn_params,
        prune_masks,
        **args,
    ):
        if bn_params is not None and len(bn_params) != 0:
            self.bn_params_exist = True
        else:
            self.bn_params_exist = False

        # Argument sanity check
        if not 0 < args['ar'] < 1:
            raise ValueError(f"given adaptivity rate {args['ar']} is invalid; "
                             "should be in (0, 1) (excluding endpoints)")
        if args['threshold'] < 0:
            raise ValueError(f"given threshold {args['threshold']} "
                             "is invalid; should be > 0")

        self.weight_flips = [torch.zeros(i.data.shape).to(args['device'])
                             for i in binary_params]
        self.prune_masks = prune_masks
        if self.bn_params_exist:
            self._adam = Adam(bn_params, lr=args['adam_lr'][0])
            start_scale = 1
            end_scale = args['adam_lr'][1] / args['adam_lr'][0]
            delta_scale = start_scale - end_scale
            self._scheduler_adam = torch.optim.lr_scheduler.LambdaLR(
                self._adam,
                lambda step: (start_scale
                              - step / args['total_iters'] * delta_scale),
                last_epoch=-1
            )

        defaults = dict(adaptivity_rate=args['ar'], threshold=args['threshold'])
        super(Bop, self).__init__(
            binary_params, defaults
        )

    def step(
        self, closure: Optional[Callable[[], float]] = ...,
        ar=None, threshold=None
    ):
        if self.bn_params_exist:
            self._adam.step()
            self._scheduler_adam.step()

        for i, group in enumerate(self.param_groups):
            params = group["params"]
            y = group["adaptivity_rate"]
            t = group["threshold"]

            if ar is not None:
                y = ar
            if threshold is not None:
                t = threshold

            for param_idx, p in enumerate(params):
                grad = p.grad.data
                state = self.state[p]

                if "moving_average" not in state:
                    m = state["moving_average"] = torch.clone(grad).detach()
                    m.add_(grad.mul(y))
                else:
                    m: Tensor = state["moving_average"]

                    m.mul_((1 - y))
                    m.add_(grad.mul(y))
                mask = (m.abs() >= t) * (m.sign() == p.sign())
                self.weight_flips[param_idx].add_(mask.double())
                mask = mask.double() * -1
                mask[mask == 0] = 1

                state["flips"] = (mask == -1).sum().item()
                p.data.mul_(mask)

    def zero_grad(self) -> None:
        super().zero_grad()
        if self.bn_params_exist:
            self._adam.zero_grad()

    def set_ar(self, ar):
        for i, group in enumerate(self.param_groups):
            group["adaptivity_rate"] = ar

    def decay_ar(self, decay_ratio):
        for i, group in enumerate(self.param_groups):
            # params = group["params"]
            group["adaptivity_rate"] *= decay_ratio

    def zero_weight_flips(self):
        for flip in self.weight_flips:
            flip.zero_()
