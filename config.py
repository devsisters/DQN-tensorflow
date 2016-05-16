class AgentConfig(object):
  display = False
  scale = 10000

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.99
  target_q_update_step = 1 * scale
  learning_rate = 0.00025

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  history_length = 4
  train_frequency = 4
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  _test_step = 1 * scale
  _save_step = _test_step * 5

class EnvironmentConfig(object):
  env_name = 'Breakout-v0'

  screen_width  = 84
  screen_height = 84
  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 4

class M2(DQNConfig):
  backend = 'tf'
  env_type = 'simple'
  action_repeat = 4

class M3(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 1

class M4(DQNConfig):
  backend = 'tf'
  env_type = 'simple'
  action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2
  elif FLAGS.model == 'm3':
    config = M3
  elif FLAGS.model == 'm4':
    config = M4
  else:
    config = M1

  for k, v in FLAGS.__dict__['__flags'].items():
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)

  return config
