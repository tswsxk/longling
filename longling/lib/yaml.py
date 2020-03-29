# coding: utf-8
# 2020/3/3 @ tongshiwei
import yaml
from collections import OrderedDict

from .stream import as_io

__all__ = ["FoldedString", "dump_folded_yaml", "ordered_yaml_load"]


def folded_string_representer(dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', data, style='|')


class FoldedString(str):
    pass


def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


def dump_folded_yaml(yaml_string):
    """specially designed for arch module, should not be used in other places"""
    yaml.add_representer(FoldedString, folded_string_representer)
    yaml.add_representer(OrderedDict, represent_ordereddict)
    output = ""
    for line in yaml.dump(yaml_string, default_flow_style=False).split("\n"):
        if line.endswith("|"):
            output += (line[:-1] + ">" + "\n")
        else:
            if line:
                output += (line + "\n")
    return output


def ordered_yaml_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    Examples
    -------
    .. code-block :: python

        ordered_yaml_load("path_to_file.yaml")
        OrderedDict({"a":123})
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    with as_io(stream) as stream:
        return yaml.load(stream, OrderedLoader)
