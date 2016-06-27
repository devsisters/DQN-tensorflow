import os
import pprint
import inspect

import tensorflow as tf

pp = pprint.PrettyPrinter().pprint

def class_vars(obj):
  return {k:v for k, v in inspect.getmembers(obj)
      if not k.startswith('__') and not callable(k)}

class BaseModel(object):
  """Abstract object representing an Reader model."""
  def __init__(self, config):
    self.config = config

    try:
      self._attrs = config.__dict__['__flags']
    except:
      self._attrs = class_vars(config)
    pp(self._attrs)

    self.config = config

    for attr in self._attrs:
      name = attr if not attr.startswith('_') else attr[1:]
      setattr(self, name, getattr(self.config, attr))

  @property
  def checkpoint_dir(self):
    return os.path.join('checkpoints', self.model_dir)

  @property
  def model_dir(self):
    model_dir = self.config.env_name
    for k, v in self._attrs.items():
      if not k.startswith('_') and k not in ['display']:
        model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
            if type(v) == list else v)
    return model_dir + '/'
