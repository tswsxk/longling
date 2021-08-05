# coding: utf-8
# 2021/8/4 @ tongshiwei

from longling.ML.PytorchHelper import Configuration


def test_configuration(tmpdir):
    model_dir = tmpdir.mkdir("pytorch_helper")
    cfg = Configuration(
        batch_size=64,
        hyper_params_update={"hidden_size": 256},
        model_dir=str(model_dir)
    )
    cfg.dump()
    cfg1 = Configuration.load(model_dir.join("configuration.json"))

    assert cfg.hyper_params["hidden_size"] == cfg1.hyper_params["hidden_size"]

    cfg2 = Configuration(model_dir.join("configuration.json"))
    assert cfg.hyper_params["hidden_size"] == cfg2.hyper_params["hidden_size"]
