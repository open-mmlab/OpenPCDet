from pathlib import Path
from typing import List

import yaml
from easydict import EasyDict
from runcon import Config


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('----------- %s -----------' % (key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


def dict_representer(dumper, ed: EasyDict):
    return dumper.represent_mapping('tag:yaml.org,2002:map', ed)


yaml.add_representer(EasyDict, dict_representer)


def cfg_to_yaml_file(config, cfg_file):
    with open(cfg_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def check_recursive_dict_type(cfg, type):
    if isinstance(cfg, dict):
        if not isinstance(cfg, type):
            raise ValueError(
                "The config is of type %s but was expected to be of type %s!\n%s"
                % (type(cfg), type, cfg)
            )
        for k in cfg:
            check_recursive_dict_type(cfg[k], type)


def ed_to_rc(ed_cfg: EasyDict) -> Config:
    check_recursive_dict_type(ed_cfg, EasyDict)
    rc_cfg = Config(ed_cfg)
    check_recursive_dict_type(rc_cfg, Config)
    return rc_cfg


def rc_to_ed(rc_cfg: Config) -> EasyDict:
    check_recursive_dict_type(rc_cfg, Config)
    ed_cfg = EasyDict()
    for k in rc_cfg:
        if isinstance(rc_cfg[k], Config):
            ed_cfg[k] = rc_to_ed(rc_cfg[k])
        else:
            ed_cfg[k] = rc_cfg[k]
    check_recursive_dict_type(ed_cfg, EasyDict)
    return ed_cfg


def modify_rc_cfg(cfg: Config, modify_cfgs: List[Path]) -> Config:
    from copy import deepcopy
    cfg = deepcopy(cfg)
    for m in modify_cfgs:
        m = cfg_from_yaml_file(m, EasyDict())
        cfg.rupdate(ed_to_rc(m))
    cfg = cfg.resolve_transforms()
    return cfg


def create_cfg_from_sets(
        cfg_file: Path,
        modify_cfgs: List[Path],
        set_cfgs: List[str],
        cfg: EasyDict = None,
) -> EasyDict:
    if cfg is None:
        cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    cfg = ed_to_rc(cfg)
    cfg = modify_rc_cfg(cfg, modify_cfgs)
    cfg = rc_to_ed(cfg)
    cfg_from_list(set_cfgs, cfg)
    return cfg


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
