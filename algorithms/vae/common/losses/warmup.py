from torch.optim import Optimizer


def warmup(optimizer: Optimizer, lr: float, warmup_factor: float) -> Optimizer:
    """
    Warmup optimizer learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * warmup_factor
    return optimizer
