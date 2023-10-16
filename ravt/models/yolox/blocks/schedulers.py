import torch


class StreamYOLOScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, exp_steps: int, last_epoch=-1):
        super(StreamYOLOScheduler, self).__init__(
            optimizer,
            lambda steps: pow(steps/exp_steps, 2) if steps < exp_steps else 0.05,
            last_epoch=last_epoch
        )


class MSCAScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, exp_steps: int, last_epoch=-1):
        def lr(steps: int) -> float:
            steps_warmup = exp_steps
            steps_1 = exp_steps * 3
            steps_2 = exp_steps * 10
            if steps < steps_warmup:
                return steps / steps_warmup
            elif steps < steps_1:
                return 1 - 0.9 * (steps - steps_warmup) / (steps_1 - steps_warmup)
            elif steps < steps_2:
                return 0.1 - 0.05 * (steps - steps_1) / (steps_2 - steps_1)
            else:
                return 0.05
        super(MSCAScheduler, self).__init__(
            optimizer,
            lr,
            last_epoch=last_epoch
        )
