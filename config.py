class AgentConfig(object):
  # memory
  memory_size = 1000000

  # DQN
  backend = 'tf' # ['tf', 'neon']
  env_type = 'simple' # ['simple', 'detail']
  action_repeat = 4
  learn_start = 50000.
  test_step = 10000

  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.99
  max_step = 50000000
  target_q_update_step = 10000
  learning_rate = 0.00025

  ep_start = 1.
  ep_end = 0.1
  ep_end_t = memory_size
  history_length = 4

class EnvironmentConfig(object):
  env_name = 'Breakout-v0'

  screen_width  = 84
  screen_height = 84

  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  pass

def get_config(FLAGS):
  if FLAGS.model == 'nature':
    config = DQNConfig
  else:
    raise ValueError(" [!] Invalid model: %s", FLAGS.model)
  return config
