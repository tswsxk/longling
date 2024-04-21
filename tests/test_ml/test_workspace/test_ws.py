# coding: utf-8
# 2022/5/12 @ tongshiwei

import pytest
from longling.ml import ws
from longling import as_out_io, path_append


@pytest.fixture
def fs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    with as_out_io(path_append(root, "train.csv")) as wf:
        wf.write("hello world")

    return {
        "root": root,
        "data": (root / "data").mkdir(),
        "model": (root / "model").mkdir(),
    }


@pytest.fixture
def ws_config_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("ws_config")


def test_ws(fs, ws_config_dir):
    wsm = ws.use(
        "ws",
        rfp=str(fs["root"]),
        space_dir=ws_config_dir,
        from_fs=True,
        skip_files=False,
        file_as_f=True,
    )
    assert wsm["train.csv"].ntype == "f"
    wsm.mkdir("space")
    wsm.add_space_to(ws.create_workspace("model1", "simple"), "space")
    assert "model1" in wsm["space"]

    ws.create_workspace("model2", "simple").add_to(wsm["space"])

    with pytest.raises(FileExistsError):
        wsm["space"].add_child(ws.create_workspace("model2", "simple"))

    with pytest.raises(TypeError):
        wsm.add_space_to(ws.create_workspace("model3", "simple"), 3)

    wsm.add_space_to(ws.create_workspace("modelx", "simple"), wsm["space"], index="modelx")
    wsm.mkdir("space/model2", ignore_existed=True)

    wsm["space/model2"].pointer = 1
    with pytest.raises(TypeError):
        print(wsm["space/model2"].ntype)

    with pytest.raises(TypeError):
        print(wsm["space/model2"].fp)

    assert "space/model3" not in wsm

    wsm.mkdir("space/model3", recursive=True)
    wsm["space/model2"].point_to(wsm["space/model3"])

    wsm["space/model2"].mount(pointer="../data/test.csv")
    wsm["space/model2"].pws()

    with pytest.raises(ValueError, match=".*"):
        wsm.rm(wsm["space/model2"])

    wsm.rm("space/model2", recursive=True)

    with pytest.raises(TypeError, match=".*"):
        wsm.rm(1)

    with pytest.raises(TypeError, match=".*"):
        wsm.mv(1, "space/model2")

    with pytest.raises(TypeError, match=".*"):
        wsm.mv("space", 2)

    with pytest.raises(TypeError, match=".*"):
        wsm.cp(1, "space/model2")

    with pytest.raises(TypeError, match=".*"):
        wsm.cp("space", 2)

    ws.commit()

    with ws.use_space("ws", space_dir=ws_config_dir) as wsm:
        wsm.ll()
        wsm.pws("space")

    wsm = ws.use("ws", space_dir=ws_config_dir, force_reinit=True)
    wsm.ll()
    assert wsm.root.fp != fs["root"]
    wsm.mount()
