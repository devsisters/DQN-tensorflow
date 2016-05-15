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
    self.checkpoint_dir = "checkpoints"

    try:
      self._attrs = config.__dict__['__flags']
    except:
      self._attrs = class_vars(config)
    pp(self._attrs)

    self.config = config

    for attr in self._attrs:
      setattr(self, attr, getattr(self.config, attr))

  def save_model(self, step=None):
    self.saver = tf.train.Saver(max_to_keep=10)

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__

    checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess, 
        os.path.join(checkpoint_dir, model_name), global_step=step)

  def load_model(self):
    self.saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(checkpoint_dir, ckpt_name)
      self.saver.restore(self.sess, fname)
      print(" [*] Load SUCCESS: %s" % fname)
      return True
    else:
      print(" [!] Load failed...")
      return False

  @property
  def model_dir(self):
    model_dir = self.config.env_name
    for k, v in self._attrs.items():
      model_dir += "/%s:%s" % (k, ",".join([str(i) for i in v])
          if type(v) == list else v)
    return model_dir
