from configparser import ConfigParser
import os 

def init_config(config_path=None) -> ConfigParser:
    config = ConfigParser()
    config.read(config_path)

    return config