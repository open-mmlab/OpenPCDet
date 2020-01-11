REGISTERED_DATASET_DICT = {}


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_DICT
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_DICT, f'exist class: {REGISTERED_DATASET_DICT}'
    REGISTERED_DATASET_DICT[name] = cls
    return cls


def get_dataset_class(name):
    global REGISTERED_DATASETS_DICT
    assert name in REGISTERED_DATASETS_DICT, f'available class: {REGISTERED_DATASET_DICT}'
    return REGISTERED_DATASET_DICT[name]
