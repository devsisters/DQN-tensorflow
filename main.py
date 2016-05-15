import tensorflow as tf

from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

flags = tf.app.flags
flags.DEFINE_string('model', 'nature', 'Type of model')
FLAGS = flags.FLAGS

def main(_):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    config = get_config(FLAGS) or FLAGS

    if config.env_type == 'simple':
      env = SimpleGymEnvironment(config.env_name, config)
    else:
      env = GymEnvironment(env_name=config.env_name,
                          screen_width=config.screen_width,
                          screen_height=config.screen_height,
                          action_repeat=config.action_repeat,
                          random_start=config.random_start)
    agent = Agent(config, env, sess)

    agent.train()

if __name__ == '__main__':
  tf.app.run()
