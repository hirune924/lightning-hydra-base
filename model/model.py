from utils.utils import load_obj

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

def get_model(cfg):
    model = load_obj(cfg.class_name)
    model = model(**cfg.params)

    return model