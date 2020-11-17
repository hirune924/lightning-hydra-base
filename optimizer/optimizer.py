from utils.utils import load_obj

def get_optimizer(model_params, cfg):
    optimizer = load_obj(cfg.class_name)
    optimizer = optimizer(model_params, **cfg.params)

    return optimizer