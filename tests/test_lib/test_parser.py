# coding: utf-8
# 2019/12/17 @ tongshiwei

import pytest
import json
import yaml
import toml
from longling import wf_open, path_append
from longling.lib.parser import ConfigurationParser, ParserGroup
from longling.lib.parser import get_parsable_var, load_configuration, Configuration


class DemoConfiguration(Configuration):
    a = 1
    b = 2


def test_get_parsable_var():
    assert set(get_parsable_var(DemoConfiguration, parse_exclude={'b'})) == {'a'}


@pytest.mark.parametrize("file_format", ["json", "toml", "yaml"])
def test_load_configuration_json(tmpdir, file_format):
    configuration = {
        "id": "12345",
        "name": "test_config"
    }

    filename = path_append(tmpdir, "test_config.%s" % file_format)

    with wf_open(filename) as wf:
        if file_format == "json":
            json.dump(configuration, wf)
        elif file_format == "toml":
            toml.dump(configuration, wf)
        elif file_format == "yaml":
            yaml.dump(configuration, wf)

    with open(filename) as f:
        _c = load_configuration(f, file_format=file_format)
        assert _c["id"] == "12345"
        assert _c["name"] == "test_config"


@pytest.mark.parametrize("file_format", ["json", "toml", "yaml"])
def test_configuration(tmpdir, file_format):
    _config = DemoConfiguration()

    assert _config.class_var == DemoConfiguration.vars()
    assert _config.parsable_var == DemoConfiguration.pvars()

    filename = path_append(tmpdir, "test_config.%s" % file_format, to_str=True)

    _config.b = 4
    _config.dump(filename, override=True, file_format=file_format)
    _config.dump(filename, override=False, file_format=file_format)

    _config = DemoConfiguration.load(filename, file_format=file_format)

    assert "a" in _config
    assert _config["a"] == 1 and _config.b == 4

    print(_config)

    assert len(_config.items()) == 2


def test_parser():
    parser = ConfigurationParser(DemoConfiguration)

    pg = ParserGroup({"train": parser, "test": parser})

    pg.print_help()
