from .downstream_mlp import DownstreamMLP


def get_base_model(cfg: dict):
    model = DownstreamMLP(**cfg["args"])

    return model
