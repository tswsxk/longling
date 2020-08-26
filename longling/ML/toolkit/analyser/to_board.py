# coding: utf-8
# 2020/5/10 @ tongshiwei

from longling import loading
from tensorboardX import SummaryWriter

__all__ = ["to_board"]

from .utils import get_by_key


def to_board(src, board_dir, global_step_field, *scalar_fields):
    with SummaryWriter(board_dir) as sw:
        for line in loading(src):
            for scalar_field in scalar_fields:
                sw.add_scalar(tag=scalar_field, scalar_value=get_by_key(line, scalar_field),
                              global_step=int(get_by_key(line, global_step_field)))
