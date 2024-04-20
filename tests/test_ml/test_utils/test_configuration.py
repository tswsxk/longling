# coding: utf-8
# 2021/6/28 @ tongshiwei

import pytest
from longling import path_append
from longling.ml.utils import Configuration, ConfigurationParser, directory_check


class DemoConfiguration(Configuration):
    optimizer = "adam"
    optimizer_params = {"lr": 0.01, "weight_decay": 0.1}
    lr = 0.01


def test_configuration():
    _config = DemoConfiguration(
        optimizer_params_update={"lr": 0.001},
        caption="model1",
        workspace="$caption",
        dataset="abc",
    )

    directory_check(_config)

    assert _config.optimizer_params["lr"] == 0.001 and _config.optimizer_params["weight_decay"] == 0.1
    assert _config.var2val("$dataset") == "abc"

    _config.update(optimizer_params_update={"lr": 0.1})
    assert _config.optimizer_params["lr"] == 0.1 and _config.optimizer_params["weight_decay"] == 0.1


@pytest.mark.parametrize("file_format", ["json", "toml", "yaml"])
def test_configuration_format(tmpdir, file_format):
    _config = DemoConfiguration()

    filename = path_append(tmpdir, "test_config.%s" % file_format, to_str=True)

    _config.dump(filename, file_format=file_format)

    _config = DemoConfiguration.load(filename, file_format=file_format)

    assert _config["optimizer"] == "adam" and _config.lr == 0.01

    _config = DemoConfiguration(params_path=filename, params_kwargs={'file_format': file_format})

    assert _config["optimizer"] == "adam" and _config.lr == 0.01


def test_parser():
    parser = ConfigurationParser(DemoConfiguration)
    c = parser("--lr float(0.1)")
    assert c["lr"] == 0.1
