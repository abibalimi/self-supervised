#!/usr/bin/env python3

import torch

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, weight_decay=0.0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get gradients
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LARS does not support sparse gradients.')

                # Compute layer-wise learning rate
                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(grad)
                trust_ratio = group['trust_coefficient'] * weight_norm / (grad_norm + 1e-8)

                # Scale learning rate
                scaled_lr = group['lr'] * trust_ratio

                # Update gradients
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(grad)
                    grad = buf

                # Update parameters
                p.data.add_(grad, alpha=-scaled_lr)

        return loss