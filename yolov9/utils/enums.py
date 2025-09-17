from enum import Enum

import yaml


def _represent_enum(dumper: yaml.BaseDumper, data: Enum):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.value)


class ModelType(Enum):
    SINGLE = "single"
    DUAL = "dual"
    TRIPLE = "triple"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


# NOTE: Currently, only supports saving
# Support custom types in yaml for dump
yaml.add_representer(ModelType, _represent_enum, yaml.SafeDumper)
