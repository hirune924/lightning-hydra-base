from utils.utils import load_obj

def get_loss(cfg):
    loss = load_obj(cfg.class_name)
    loss = loss(**cfg.params)

    return loss