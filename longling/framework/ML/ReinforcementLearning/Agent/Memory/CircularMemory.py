# coding:utf-8
# created by tongshiwei on 2018/11/8
from __future__ import absolute_import
import random

from longling.framework.ML.ReinforcementLearning.Agent.Memory.Memory import Memory


class CircularMemory(Memory):
    def add(self, new_memory):
        if len(self._memory) == self.max_size:
            self._memory[self.top] = new_memory
            self.top = (self.top + 1) % self.max_size
        elif len(self._memory) < self.max_size:
            self._memory.append(new_memory)
        else:
            self.logger.warning("detect the size of current size bigger than max_size, maybe existing error somewhere")
            self._memory = random.sample(self._memory, self.max_size)
