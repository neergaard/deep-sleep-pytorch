import os
import time

import yaml
from dotmap import DotMap


def get_config_from_yaml(yaml_file):
    """
    Get the config from a yaml file
    :param yaml_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config yaml file provided
    with open(yaml_file, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(yaml_file):
    config, _ = get_config_from_yaml(yaml_file)

    return config
