import yaml

from pathlib import Path


class MergedConfig:
    def __init__(self) -> None:
        pass


def merge_class_attrs(ob1, ob2):
    ob1.__dict__.update(ob2.__dict__)
    return ob1


def save_cfgs(cfgs, fpath):
    ''' save config'''
    Path(fpath).mkdir(parents=True, exist_ok=True)

    with open(f"{fpath}/config.yaml", 'w') as f:
        for cfg_type in cfgs:
            yaml.dump({cfg_type: cfgs[cfg_type].__dict__}, f, default_flow_style=False)
