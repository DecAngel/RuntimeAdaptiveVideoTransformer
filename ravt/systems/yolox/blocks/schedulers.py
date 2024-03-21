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
            intervals = [0, exp_steps, exp_steps*2, exp_steps*3, exp_steps*6, exp_steps*1000]
            values = [0.05, 0.2, 1, 0.2, 0.05]
            for left, right, v in zip(intervals[:-1], intervals[1:], values):
                if left <= steps <= right:
                    return v

        super(MSCAScheduler, self).__init__(
            optimizer,
            lr,
            last_epoch=last_epoch
        )
