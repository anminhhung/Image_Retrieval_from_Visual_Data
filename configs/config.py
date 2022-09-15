from configs.dolg.dolg_b5_step3 import cfg_b5
from configs.dolg.dolg_b6_step3 import cfg_b6
from configs.dolg.dolg_b7_step1 import cfg_b7_step1
from configs.dolg.dolg_b7_step2 import cfg_b7_step2
from configs.dolg.dolg_b7_step3 import cfg_b7_step3
from configs.swin.swin_224_b3 import cfg_swin_224_b3
from configs.swin.swin_224_b5 import cfg_swin_224_b5
from configs.swin.swin_224_b6 import cfg_swin_224_b6

def init_config(config_path=None):
    config = None
    if "dolg_b5_step3" in config_path:
        config = cfg_b5
    elif "dolg_b6_step3" in config_path:
        config = cfg_b6
    elif "dolg_b7_step1" in config_path:
        config = cfg_b7_step1
    elif "dolg_b7_step2" in config_path:
        config = cfg_b7_step2
    elif "dolg_b7_step3" in config_path:
        config = cfg_b7_step3
    elif "swin_224_b3" in config_path:
        config = cfg_swin_224_b3
    elif "swin_224_b5" in config_path:
        config = cfg_swin_224_b5
    elif "swin_224_b6" in config_path:
        config = cfg_swin_224_b6

    return config