from .combine_loss import CombineLoss


def get_loss_fn(cfg: dict):
    return CombineLoss(**cfg["args"])
