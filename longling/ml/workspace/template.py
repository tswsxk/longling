# coding: utf-8
# 2022/5/9 @ tongshiwei

# from base import SpaceTreeNode
from .base import SpaceTreeNode


class TemplateSpace(SpaceTreeNode):
    pass


class SimpleTemplateSpace(TemplateSpace):
    def __init__(self, name, input_pointer=None, output_pointer=None, model_pointer=None, tmp_pointer=None):
        """

        Parameters
        ----------
        name
        input_pointer
        output_pointer
        model_pointer
        tmp_pointer

        Examples
        ---------
        >>> import os
        >>> os.path.sep = "/"
        >>> space = SimpleTemplateSpace(
        ...     "sts",
        ...     input_pointer="../data/input",
        ...     output_pointer="../data/output",
        ...     model_pointer="../data/model",
        ...     tmp_pointer="~/tmp"
        ... )
        >>> space.ll()
        Space[sts - (a)]: / -> /
        ------------------------
        input -> ../data/input
        output -> ../data/output
        model -> ../data/model
        tmp -> ~/tmp
        ------------------------
        """
        super(SimpleTemplateSpace, self).__init__(name)
        self.mkdir("input", pointer=input_pointer)
        self.mkdir("output", pointer=output_pointer)
        self.mkdir("model", pointer=model_pointer)
        self.mkdir("tmp", pointer=tmp_pointer)
