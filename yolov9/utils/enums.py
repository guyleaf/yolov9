from enum import Enum

import yaml

TAG_PREFIX = "!yolov9/utils/enums"
STR_ENUM_TAG = f"{TAG_PREFIX}/StrEnum"


class StrEnum(str, Enum):
    """
    Enum where members are also (and must be) strings

    Modified from CPython
    """

    def __new__(cls, *values):
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError("too many arguments for str(): %r" % (values,))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError("%r is not a string" % (values[0],))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError("encoding must be a string, not %r" % (values[1],))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()

    def __str__(self):
        return self.value


class ModelType(StrEnum):
    SINGLE = "single"
    DUAL = "dual"
    TRIPLE = "triple"


def _represent_str_enum(dumper: yaml.SafeDumper, data: StrEnum):
    cls = type(data)
    return dumper.represent_scalar(f"{STR_ENUM_TAG}:{cls.__name__}", data.value)


def _construct_str_enum(loader: yaml.SafeLoader, suffix: str, node: yaml.Node):
    # suffix_tag = node.tag[len(f"{STR_ENUM_TAG}:") :]
    value: str = loader.construct_yaml_str(node)
    return globals()[suffix](value)


# Support custom types in yaml
yaml.add_multi_representer(StrEnum, _represent_str_enum, yaml.SafeDumper)
yaml.add_multi_constructor(f"{STR_ENUM_TAG}:", _construct_str_enum, yaml.SafeLoader)
