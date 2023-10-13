import torch


class StreamYOLOScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, exp_steps: int, last_epoch=-1):
        super(StreamYOLOScheduler, self).__init__(
            optimizer,
            lambda steps: pow(steps/exp_steps, 2) if steps < exp_steps else 0.05,
            last_epoch=last_epoch
        )


class SwinScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, exp_steps: int, last_epoch=-1):
        def lr(steps: int) -> float:
            stage_0 = exp_steps
            stage_1 = 10 * exp_steps
            stage_2 = 13 * exp_steps
            if steps < stage_0:
                return steps / stage_0
            elif steps < stage_1:
                return 1
            elif steps < stage_2:
                return 1 - 0.9 * (steps - stage_1) / (stage_2 - stage_1)
            else:
                return 0.1

        super(SwinScheduler, self).__init__(
            optimizer,
            lr_lambda=lr,
            last_epoch=last_epoch
        )
