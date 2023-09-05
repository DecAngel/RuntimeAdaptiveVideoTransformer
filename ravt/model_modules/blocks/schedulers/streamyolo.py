import torch


class StreamYOLOLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, exp_steps: int, last_epoch=-1):
        super(StreamYOLOLR, self).__init__(
            optimizer,
            lambda steps: pow(steps/exp_steps, 2) if steps < exp_steps else 0.05,
            last_epoch=last_epoch
        )
