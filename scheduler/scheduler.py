from utils.utils import load_obj

def get_scheduler(optimizer, cfg):
    scheduler = load_obj(cfg.class_name)
    scheduler = scheduler(optimizer, **cfg.params)
    return scheduler