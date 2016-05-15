import random
from collections import deque

class Memory(object):
  def __init__(self, config):
    self.memory_size = config.memory_size
    self.batch_size = config.batch_size
    self.m = deque()

  def add(self, *args):
    self.m.append(list(args))

    if len(self.m) > self.memory_size:
      self.m.popleft()

  def sample(self):
    m_length = len(self.m)

    if m_length >= self.batch_size:
      return self.deserialize(random.sample(self.m, self.batch_size))
    else:
      return self.deserialize(random.sample(self.m, m_length))

  def deserialize(self, ms):
    screens = [m[0] for m in ms]
    rewards = [m[1] for m in ms]
    actions = [m[2] for m in ms]
    next_screens = [m[3] for m in ms]
    terminals = [m[4] for m in ms]

    return screens, actions, rewards, next_screens, terminals
