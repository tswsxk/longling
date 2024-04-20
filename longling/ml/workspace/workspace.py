# coding: utf-8
# 2022/5/8 @ tongshiwei
import warnings
import os
import joblib
from typing import Union
from longling import path_append, PATH_TYPE
from contextlib import contextmanager
from .base import SpaceTreeNode
from .template import SimpleTemplateSpace

# from base import SpaceTreeNode
# from template import SimpleTemplateSpace

TEMPLATE_MAPPING = {
    "simple": SimpleTemplateSpace,
}
DEFAULT_SPACE_DIR = "~/.lws"


def create_workspace(name, template: str = None, *args, **kwargs) -> SpaceTreeNode:
    """

    Parameters
    ----------
    name
    template
    args
    kwargs

    Returns
    -------
    >>> import  os
    >>> os.path.sep = "/"
    >>> space = create_workspace("simple", "simple")
    >>> space.ls()
    input  output  model  tmp
    >>> space = create_workspace("ws")
    >>> space.ls()
    (null space)
    """
    if template is not None:
        return TEMPLATE_MAPPING[template](name, *args, **kwargs)
    else:
        return SpaceTreeNode(name, *args, **kwargs)


class WorkSpaceManager(object):
    def __init__(
            self, name, rfp=None, space_dir=DEFAULT_SPACE_DIR,
            initial_space=True,
            from_fs=False, skip_files=False, file_as_f=False
    ):
        """

        Parameters
        ----------
        name
        rfp
        space_dir
        initial_space
        from_fs
        skip_files
        file_as_f

        Examples
        --------
        >>> import os
        >>> os.path.sep ="/"
        >>> wsm = WorkSpaceManager("ws", "../data")
        >>> wsm
        WorkSpace[ws]
        >>> wsm.la()
        (f) / -> ../data
        >>> _ =wsm.mkdir("data")
        >>> _ = wsm.mkn("model")
        >>> wsm.ls()
        data  model
        >>> wsm.rm("data")
        >>> _ = wsm.mkdir("space/model", index="model", recursive=True)
        >>> wsm.mv("space/model", "space/model1")
        >>> "model" in wsm.index
        True
        >>> wsm.mkl("space", "../space")
        >>> wsm.root
        Space[ws - (f)]: / -> ../data
        >>> "model1" in wsm.cs("space")
        True
        >>> for ntype, name, sp, fp in wsm.walk():
        ...     pass
        >>> wsm.mki("space", "space")
        >>> "space" in wsm.index
        True
        """
        self.config_path = path_append(space_dir, name)
        self.name = name
        if initial_space:
            self.space_tree: SpaceTreeNode = SpaceTreeNode(name, pointer=rfp)
            if from_fs:
                self.from_fs(rfp, skip_files, file_as_f)
        else:
            self.space_tree: SpaceTreeNode = joblib.load(self.config_path)

    def walk(self):
        return self.space_tree.walk()

    def la(self, path=None):
        for ntype, _, sp, fp in self[path].walk():
            print("(%s) %s -> %s" % (ntype, sp, fp))

    def __repr__(self):
        return "WorkSpace[%s]" % self.name

    def from_fs(self, rfp=None, skip_files=False, file_as_f=False, inc=False):
        self.space_tree.from_fs(rfp, skip_files, file_as_f, inc)

    @property
    def root(self):
        return self.space_tree

    @property
    def index(self):
        return self.space_tree.index

    def __contains__(self, item):
        return item in self.space_tree

    def __getitem__(self, item):
        return self.space_tree[item]

    def add_space_to(self, src: SpaceTreeNode, tar: (PATH_TYPE, SpaceTreeNode), index=None):
        if isinstance(tar, PATH_TYPE):
            self.space_tree[tar].add_child(src)
        elif isinstance(tar, SpaceTreeNode):
            tar.add_child(src)
        else:
            raise TypeError()
        if index is not None:
            self.mki(src, index)

    def pws(self, path):
        self.space_tree[path].pws()

    def ls(self, path=None):
        self.space_tree[path].ls()

    def ll(self, path=None):
        self.space_tree[path].ll()

    def cs(self, path):
        return self.space_tree[path]

    def mkn(self, name, pointer=None, path=None, index=None, recursive=False, ignore_existed=False):
        return self.space_tree.mkn(
            name,
            pointer,
            path,
            index,
            recursive,
            ignore_existed
        )

    def mkdir(self, name, pointer=None, path=None, index=None, recursive=False, ignore_existed=False):
        return self.space_tree.mkdir(
            name,
            pointer,
            path,
            index,
            recursive,
            ignore_existed
        )

    def mkl(self, path, pointer):
        self.space_tree.mkl(path, pointer)

    def mount(self, path=None, pointer=None):
        self.space_tree.mount(path, pointer)

    def mki(self, path, index_name):
        self.space_tree.mki(path, index_name)

    def rm(self, name, recursive=False, force=False, del_from_index=True):
        self.space_tree.rm(name, recursive, force, del_from_index)

    def mv(self, src, tar):
        self.space_tree.mv(src, tar)

    def cp(self, src, tar):
        return self.space_tree.cp(src, tar)

    def commit(self):
        self.save()

    def save(self, filepath=None):
        filepath = self.config_path if filepath is None else filepath
        joblib.dump(self.space_tree, str(filepath))


wsm: Union[WorkSpaceManager, None] = None


def use(name, rfp=None, space_dir=DEFAULT_SPACE_DIR, force_reinit=False,
        from_fs=False, skip_files=True, file_as_f=False) -> WorkSpaceManager:
    global wsm

    if os.path.exists(path_append(space_dir, name)):
        if not force_reinit:
            wsm = WorkSpaceManager(name, space_dir=space_dir, initial_space=False)
            return wsm
        else:
            warnings.warn("WorkSpace existed but is required to reinit, overridden")
    wsm = init(name, rfp, space_dir, from_fs=from_fs, skip_files=skip_files, file_as_f=file_as_f)
    warnings.warn(
        "WorkSpace %s not existed, created, configuration is located at %s (abspath: %s). "
        "To avoid this warning, use init() to create space before using."
        % (name, wsm.config_path, os.path.abspath(wsm.config_path))
    )

    return wsm


def init(name, rfp, space_dir=DEFAULT_SPACE_DIR, from_fs=False, skip_files=True, file_as_f=False):
    global wsm
    wsm = WorkSpaceManager(
        name, rfp, space_dir, initial_space=True,
        from_fs=from_fs, skip_files=skip_files, file_as_f=file_as_f
    )
    return wsm


def commit():
    global wsm
    assert wsm is not None
    wsm.commit()


@contextmanager
def use_space(name, rfp=None, space_dir=DEFAULT_SPACE_DIR, force_reinit=False,
              from_fs=False, skip_files=True, file_as_f=False) -> WorkSpaceManager:
    global wsm
    wsm = use(name, rfp, space_dir, force_reinit, from_fs, skip_files, file_as_f)
    yield wsm
    commit()
