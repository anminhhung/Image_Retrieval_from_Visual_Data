from configparser import ConfigParser
import os 
from configs.dolg.dolg_b5_step3 import cfg_b5
from configs.dolg.dolg_b6_step3 import cfg_b6
from configs.dolg.dolg_b7_step1 import cfg_b7_step1
from configs.dolg.dolg_b7_step2 import cfg_b7_step2
from configs.dolg.dolg_b7_step3 import cfg_b7_step3

def init_config(config_path=None) -> ConfigParser:

    # config = ConfigParser()
    # config.read(config_path)
    config = None
    if "dolg_b5_step3" in config_path:
        config = cfg_b5
    elif "dolg_b6_step3":
        config = cfg_b6
    elif "dolg_b7_step1":
        config = cfg_b7_step1
    elif "dolg_b7_step2":
        config = cfg_b7_step2
    elif "dolg_b7_step3":
        config = cfg_b7_step3
    return config