class AgentConfig(object):
  scale = 10000
  display = False

  max_step = 8000 * scale

  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.99
  target_q_update_step = 4 * scale
  learning_rate = 0.0007

  decay = 0.99
  epsilon = 0.1
  momentum = 0.0
  beta = 0.01

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = 400 * scale

  history_length = 4
  batch_size = 32
  train_frequency = batch_size
  learn_start = batch_size

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  _test_step = 0.5 * scale

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
  action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
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
