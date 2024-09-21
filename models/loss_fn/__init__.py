from .combine_loss import CombineLoss


def get_loss(cfg: dict):
    return CombineLoss(**cfg["args"])
