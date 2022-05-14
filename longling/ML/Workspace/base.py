# coding: utf-8
# 2022/5/9 @ tongshiwei

import itertools as it
import os
from typing import Union
from longling import path_append, PATH_TYPE
from copy import deepcopy


class SpaceNode(object):
    def __init__(self, *args, **kwargs):
        self.pointer = None
        self.name = None
        self.parent = None

    @property
    def fp(self):
        raise NotImplementedError

    @property
    def sp(self):
        raise NotImplementedError

    def point_to(self, *args, **kwargs):
        raise NotImplementedError

    def add_child(self, child, ignore_existed=False):
        raise NotImplementedError

    def mkl(self, *args, **kwargs):
        raise NotImplementedError

    def mkn(self, name, pointer=None, path=None, index=None, recursive=False, ignore_existed=False):
        raise NotImplementedError

    def mkdir(self, name, pointer=None, path=None, index=None, recursive=False, ignore_existed=False):
        return self.mkn(
            name,
            pointer,
            path,
            index,
            recursive,
            ignore_existed
        )

    @classmethod
    def norm_name(cls, name):
        path = path_append(name)
        name = path.name
        parts = path.parts[:-1]
        parts = None if not parts else parts
        return parts, name

    def ls(self, *args, **kwargs):
        raise NotImplementedError

    def ll(self, *args, **kwargs):
        raise NotImplementedError

    def pws(self, *args, **kwargs):
        raise NotImplementedError


class SpaceTreeNode(SpaceNode):
    def __init__(self, name: str, pointer: (PATH_TYPE, SpaceNode) = None):
        r"""

        Two mode:

        node -> space node
        node -> file system node

        Parameters
        ----------
        name: str
            Node name
        pointer: PATH_TYPE
            the pointer pointing to real file location

        Methods
        --------
        mount:
            link the space node with specific file node


        Examples
        -------
        >>> import  os
        >>> os.path.sep = "/"
        >>> SpaceTreeNode("sher")
        Space[sher - (a)]: / -> /
        >>> SpaceTreeNode("sher", "../data/")
        Space[sher - (f)]: / -> ../data/
        >>> root = SpaceTreeNode("sher", SpaceTreeNode("lock", "../model"))
        >>> root
        Space[sher - (s)]: / -> ../model
        >>> root.sp
        '/'
        >>> root.fp
        '../model'
        >>> root.ntype
        's'
        >>> root_space = SpaceTreeNode("sher", "../data")
        >>> root_space.mkdir("data")
        Space[data - (a)]: \data -> ..\data\data
        >>> root_space.mkdir("model", pointer="../data/model")
        Space[model - (f)]: \model -> ../data/model
        >>> root_space["data"].ls()
        (null space)
        >>> root_space["data"].ll()
        Space[data - (a)]: \data -> ..\data\data
        ----------------------------------------
        (null space)
        ----------------------------------------
        >>> root_space["/data"].mkn("train.csv").mount()
        >>> root_space["/data"].mkn("test.csv").mount()
        >>> root_space.mki("/data/train.csv", "train.csv")
        >>> root_space.mki("/data/test.csv", "test.csv")
        >>> root_space.index["train.csv"]
        Space[train.csv - (f)]: \data\train.csv -> ..\data\data\train.csv
        >>> root_space.index["test.csv"]
        Space[test.csv - (f)]: \data\test.csv -> ..\data\data\test.csv
        >>> _ = root_space.rm("/data/train.csv")
        >>> root_space.index
        {'test.csv': Space[test.csv - (f)]: \data\test.csv -> ..\data\data\test.csv}
        >>> root_space["/data"].reset_pointer()
        >>> data2 = root_space.mkdir("/data/data2")
        >>> _ = root_space.cp("/data/data2", "/data/data1")
        >>> "data/data2" in root_space
        True
        >>> _ = root_space.cp(data2, "/data/data3")
        >>> "data/data3" in root_space
        True
        >>> _ = root_space.rm(root_space["data/data3"])
        >>> "data/data3" in root_space
        False
        >>> _ = root_space.mv("data/data2", "/")
        >>> _ = root_space.cp("data/data1", root_space)
        >>> _ = root_space.mv("/data2", root_space["data"])
        """
        super(SpaceTreeNode, self).__init__()
        self.name = name
        self.parent: Union[SpaceNode, None] = None
        self.pointer = pointer
        self.children = {}
        self.index = {}
        self._rindex = []

    def from_fs(self, rfp: PATH_TYPE = None, skip_files=True, file_as_f=False, ignore_existed=False):
        if rfp is not None:
            self.pointer = rfp

        rfp = self.pointer
        assert isinstance(rfp, PATH_TYPE)
        for root, dirs, files in os.walk(rfp):
            root = os.path.relpath(root, rfp)
            for _dir in dirs:
                self.mkn(_dir, path=root, ignore_existed=ignore_existed)
            if not skip_files:
                for _file in files:
                    pointer = path_append(root, _file) if file_as_f else None
                    self.mkn(_file, pointer=pointer, path=root, ignore_existed=ignore_existed)

    def add_rindex(self, space, index):
        self._rindex.append((space, index))

    def del_from_index(self):
        for space, index in self._rindex:
            space.index.pop(index)

    def add_to(self, parent: SpaceNode):
        self.parent = parent
        parent.add_child(self)

    def add_child(self, child, ignore_existed=False):
        if child.name in self.children:
            if ignore_existed:
                return
            else:
                raise FileExistsError()
        self.children[child.name] = child
        child.parent = self

    def mki(self, src: (PATH_TYPE, SpaceNode), tar: str):
        assert tar not in self.index
        if isinstance(src, PATH_TYPE):
            self.index[tar] = self[src]
            src = self[src]
        else:
            self.index[tar] = src

        src.add_rindex(self, tar)

    def __repr__(self):
        return "Space[%s - (%s)]: %s -> %s" % (self.name, self.ntype, self.sp, self.fp)

    @property
    def ntype(self):
        if self.pointer is None:
            return "a"  # auto
        if isinstance(self.pointer, SpaceNode):
            return "s"  # space
        elif isinstance(self.pointer, PATH_TYPE):
            return "f"  # file system
        else:
            raise TypeError()

    @property
    def fp(self):
        if self.pointer is None:
            if self.parent is None:
                return "/"
            return path_append(self.parent.fp, self.name)
        elif isinstance(self.pointer, SpaceNode):
            return self.pointer.fp
        elif isinstance(self.pointer, PATH_TYPE):
            return self.pointer
        else:
            raise TypeError()

    @property
    def sp(self):
        if self.parent is None:
            return "/"
        else:
            return path_append(self.parent.sp, self.name)

    def reset_pointer(self):
        self.pointer = None

    def __contains__(self, item):
        try:
            _ = self[item]
            return True
        except KeyError:
            return False

    def __getitem__(self, item):
        if item is None:
            return self
        elif isinstance(item, PATH_TYPE):
            select_list = path_append(item).parts
        else:
            select_list = item

        if select_list[0] in {".", path_append("/", to_str=True)}:
            select_list = select_list[1:]
        node = self
        for select in select_list:
            node = node.children[select]
        return node

    def mkn(self, name, pointer=None, path=None, index=None, recursive=False, ignore_existed=False):
        path = path_append(name) if path is None else path_append(path, name)
        parts, name = self.norm_name(path)
        child = SpaceTreeNode(name, pointer)

        if not recursive:
            self[parts].add_child(child, ignore_existed)
        else:
            idx = 0
            tree_cursor = self
            while idx < len(parts):
                if parts[idx] not in tree_cursor.children:
                    tree_cursor.mkn(parts[idx])
                tree_cursor = tree_cursor.children[parts[idx]]
                idx += 1
            tree_cursor.add_child(child, ignore_existed)

        if index is not None:
            self.mki(child, index)

        return child

    def mkl(self, path, pointer: (PATH_TYPE, SpaceNode)):
        self[path].pointer = pointer

    def point_to(self, pointer: (PATH_TYPE, SpaceNode)):
        self.pointer = pointer

    def mount(self, path=None, pointer: PATH_TYPE = None):
        path = "/" if path is None else path
        if pointer is None:
            self.mkl(path, self[path].fp)
        else:
            assert isinstance(pointer, PATH_TYPE)
            self.mkl(path, pointer)

    def walk(self):
        return [(self.ntype, self.name, self.sp, self.fp)] + list(
            it.chain(*[child.walk() for child in self.children.values()]))

    def ls(self):
        if not self.children:
            print("(null space)")
        else:
            print("  ".join(self.children))

    def ll(self):
        print(self)
        print("-" * len(repr(self)))
        if self.children:
            print("\n".join(["%s -> %s" % (child.name, child.fp) for child in self.children.values()]))
        else:
            print("(null space)")
        print("-" * len(repr(self)))

    def pws(self):
        print(self.sp)

    def rm(self, name: (PATH_TYPE, SpaceNode), recursive=False, force=False, del_from_index=True):
        if isinstance(name, PATH_TYPE):
            parts, name = self.norm_name(name)
            cursor = self[parts]
        elif isinstance(name, SpaceNode):
            cursor = name.parent
            name = name.name
        else:
            raise TypeError()

        assert name in cursor.children
        child = cursor.children[name]
        if del_from_index:
            child.del_from_index()
        if child.children:
            if recursive:
                for _child in list(child.children.keys()):
                    child.rm(_child, recursive=True)
            elif force:
                pass
            else:
                raise ValueError("%s is not Empty" % child)
        child = cursor.children.pop(name)
        child.parent = None
        return child

    def mv(self, src: (PATH_TYPE, SpaceNode), tar: (PATH_TYPE, SpaceNode)):
        if not isinstance(tar, (PATH_TYPE, SpaceNode)):
            raise TypeError()
        child = self.rm(src, recursive=False, del_from_index=False, force=True)
        if isinstance(tar, PATH_TYPE):
            if isinstance(tar, str) and path_append(tar[-1], to_str=True) == path_append("/", to_str=True):
                self[tar].add_child(child)
            else:
                parts, name = self.norm_name(tar)
                child.name = name
                self[parts].add_child(child)
        elif isinstance(tar, SpaceNode):
            tar.add_child(child)
        else:  # pragma: no cover
            raise TypeError()
        return child

    def cp(self, src: (PATH_TYPE, SpaceNode), tar: (PATH_TYPE, SpaceNode)):
        if isinstance(src, PATH_TYPE):
            child = deepcopy(self[src])
        elif isinstance(src, SpaceNode):
            child = deepcopy(src)
        else:
            raise TypeError()
        return self.mv(child, tar)
