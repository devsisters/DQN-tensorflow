import random
import tensorflow as tf

from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 1, 'The number of action to be repeated')

# Distributed
flags.DEFINE_string("ps_hosts", "0.0.0.0:2222", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "0.0.0.0:2223,0.0.0.0:2224", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Misc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

def main(_):
  config = get_config(FLAGS) or FLAGS
  if not FLAGS.use_gpu:
    config.cnn_format = 'NHWC'

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  server_config = tf.ConfigProto(
      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4), log_device_placement=True)

  server = tf.train.Server(cluster,
                           config=server_config,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    env = GymEnvironment(config)

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model
      agent = Agent(config, env)

    print(agent.model_dir)

    # Create a "supervisor", which oversees the training process.
    is_chief = (FLAGS.task_index == 0)
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="./logs/" + agent.model_dir,
                             init_op=agent.init_op,
                             summary_op=None,
                             saver=agent.saver,
                             global_step=agent.step_op,
                             save_model_secs=600)

    if FLAGS.is_train:
      if is_chief:
        train_or_play = agent.train_with_summary
      else:
        train_or_play = agent.train
    else:
      train_or_play = agent.play

    with sv.managed_session(server.target) as sess:
      agent.sess = sess
      agent.update_target_q_network()

      train_or_play(sv, is_chief)

  # Ask for all the services to stop.
  sv.stop()

if __name__ == '__main__':
  tf.app.run()
