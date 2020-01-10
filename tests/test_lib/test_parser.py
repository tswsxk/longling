# coding: utf-8
# 2019/12/17 @ tongshiwei

import json
from longling import wf_open, path_append
from longling.lib.parser import get_parsable_var, load_configuration_json, Configuration
from longling.lib.parser import ConfigurationParser, ParserGroup


class DemoConfiguration(Configuration):
    a = 1
    b = 2


def test_get_parsable_var():
    assert set(get_parsable_var(DemoConfiguration, parse_exclude={'b'})) == {'a'}


def test_load_configuration_json(tmpdir):
    filename = path_append(tmpdir, "test_config.json")

    configuration = {
        "id": "12345",
        "name": "test_config"
    }

    with wf_open(filename) as wf:
        json.dump(configuration, wf)

    with open(filename) as f:
        _c = load_configuration_json(f)
        assert _c["id"] == "12345"
        assert _c["name"] == "test_config"


def test_configuration(tmpdir):
    _config = DemoConfiguration()

    assert _config.class_var == DemoConfiguration.vars()
    assert _config.parsable_var == DemoConfiguration.pvars()

    filename = path_append(tmpdir, "test_config.json", to_str=True)

    _config.b = 4
    _config.dump(filename)
    _config.dump(filename)

    _config = DemoConfiguration.load(filename)

    assert "a" in _config
    assert _config["a"] == 1 and _config.b == 4

    print(_config)

    assert len(_config.items()) == 2


def test_parser():
    parser = ConfigurationParser(DemoConfiguration)

    pg = ParserGroup({"train": parser, "test": parser})

    pg.print_help()
