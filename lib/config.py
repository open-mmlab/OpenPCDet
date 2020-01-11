from easydict import EasyDict
import argparse
from pathlib import Path
import yaml


def save_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            if logger is not None:
                logger.info('\n%s.%s = edict()' % (pre, key))
            else:
                print('\n%s.%s = edict()' % (pre, key))
            save_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue

        if logger is not None:
            logger.info('%s.%s: %s' % (pre, key, val))
        else:
            print('%s.%s: %s' % (pre, key, val))


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
            # handle the case when v is a string literal
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


def get_parser():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument("--train_mode", type=str, default='joint', required=False, help="specify the training mode")
    parser.add_argument("--batch_size", type=int, default=16, required=False, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=80, required=False, help="Number of epochs to train for")
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument("--extra_tag", type=str, default=None, help="extra tag for this experiment")
    parser.add_argument("--ckpt", type=str, default=None, help='checkpoint to start from')
    parser.add_argument("--pretrained_model", type=str, default=None, help='pretrained_model')
    parser.add_argument("--rpn_ckpt", type=str, default=None, help="rpn checkpoint for rcnn training")
    parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')
    parser.add_argument('--dist', action='store_true', default=False, help='whether to use distribute training')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument("--tcp_port", type=int, default=18888, help="tcp port for distrbuted training")
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--train_with_eval', action='store_true', default=False, help='whether to eval when training')
    parser.add_argument("--ckpt_save_interval", type=int, default=1, help="number of training epochs")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--max_ckpt_save_num", type=int, default=30, help="")
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument("--eval_mode", type=str, default='joint', required=False, help="specify the evaluation mode")
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--test', action='store_true', default=False, help='evaluate without ground truth')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument("--eval_tag", type=str, default=None, help="extra tag for multiple evaluation")
    parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
    parser.add_argument("--ckpt_dir", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")

    parser.add_argument('--fuse_bn', action='store_true', default=False)

    # for prepare data
    parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'nuscenes', 'waymo'])

    args = parser.parse_args()

    if args.cfg_file is not None:
        with open(args.cfg_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(config)

        for key, val in args.__dict__.items():
            config[key] = val

        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs, config)
        config.TAG = Path(args.cfg_file).stem

        config.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
        config.LOCAL_RANK = 0
        return config, args
    else:
        config = EasyDict()
        config.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
        config.LOCAL_RANK = 0
        return config, args


cfg, args = get_parser()

