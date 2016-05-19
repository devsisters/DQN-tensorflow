import random
import tensorflow as tf

from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

flags = tf.app.flags
flags.DEFINE_string('model', 'nature', 'Type of model')
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

def main(_):
  config = get_config(FLAGS) or FLAGS

  if config.env_type == 'simple':
    env = SimpleGymEnvironment(config)
  else:
    env = GymEnvironment(config)

  with tf.Session() as sess, tf.device('/cpu:0'):
    agent = Agent(config, env, sess, FLAGS.random_seed)

    if FLAGS.is_train:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
