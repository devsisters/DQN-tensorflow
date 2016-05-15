import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils import inject_summary, timeit
from .base import BaseModel
from .history import History
from .ops import linear, conv2d
from .memory import Memory

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)

    self.sess = sess
    self.env = environment
    self.history = History(config)
    self.memory = Memory(config)

    self.step_op = tf.Variable(0, trainable=False)
    self.ep_op = tf.Variable(self.ep_start, trainable=False)

    self.build_dqn()

  def train(self):
    tf.initialize_all_variables().run()

    self.update_target_q_network()
    self.load_model()

    start_step = self.step.eval()
    start_time = time.time()

    num_game = 0
    total_reward = 0.
    self.total_loss = 0.
    self.total_q = 0.
    self.update_count = 0
    ep_reward = 0.
    max_ep_reward = 0.
    min_ep_reward = 99999.

    screen, reward, action, terminal = self.env.new_random_game()

    for self.step in tqdm(range(start_step, self.max_step), ncols=100):
      action = self.perceive(screen, reward, action, terminal)

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()

        min_ep_reward = min(ep_reward, min_ep_reward)
        max_ep_reward = max(ep_reward, max_ep_reward)
        num_game += 1

        ep_reward = 0.
      else:
        screen, reward, terminal = self.env.act(action, is_training=True)
        ep_reward += reward

      if self.step > self.learn_start and self.step % self.test_step == self.test_step - 1:
        avg_reward = total_reward / self.test_step
        avg_loss = self.total_loss / self.update_count
        avg_q = self.total_q / self.update_count

        print "\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d" \
            % (avg_reward, avg_loss, avg_q, max_ep_reward, min_ep_reward, num_game)

        inject_summary(self.writer, "average reward", avg_reward, self.step)
        inject_summary(self.writer, "average loss", avg_loss, self.step)
        inject_summary(self.writer, "average q", avg_q, self.step)
        inject_summary(self.writer, "max episode reward", max_ep_reward, self.step)
        inject_summary(self.writer, "min episode reward", min_ep_reward, self.step)
        inject_summary(self.writer, "# of game", num_game, self.step)

        num_game = 0
        total_reward = 0.
        self.total_loss = 0.
        self.total_q = 0.
        self.update_count = 0
        ep_reward = 0.
        max_ep_reward = 0.
        min_ep_reward = 99999.

        self.step_op.assign(self.step).eval()
        self.save_model()
      else:
        total_reward += reward

  def perceive(self, screen, reward, action, terminal, test_ep=None):
    # reward clipping
    reward = max(self.min_reward, min(self.max_reward, reward))

    # add memory
    prev_hist = self.history.get()
    self.history.add(screen)

    if test_ep == None:
      self.memory.add(prev_hist[0], reward, action, self.history.get()[0], terminal)

    # e greedy
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randint(0, self.env.action_size - 1)
    else:
      action = self.q_action.eval({self.s_t: self.history.get()})

    if self.step > self.learn_start:
      if test_ep == None:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

    return action

  def q_learning_mini_batch(self):
    s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

    t = time.time()
    q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

    terminal = np.array(terminal) + 0.
    max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
    target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    _, loss = self.sess.run([self.optim, self.loss], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
    })

    self.total_loss += loss
    self.total_q += q_t_plus_1.mean()
    self.update_count += 1

  def build_dqn(self):
    if self.backend == 'neon':
      from neon.backends import gen_backend
      from neon.optimizers import RMSProp
      from neon.layers import Affine, Conv, GeneralizedCost
      from neon.transforms import Rectlin
      from neon.models import Model
      from neon.transforms import SumSquared

      stochastic_round = True

      self.be = gen_backend(backend = 'gpu',
          batch_size = self.batch_size,
          datatype = np.dtype('float32').type,
          stochastic_round = stochastic_round)

      self.input_shape = (self.history_length, args.screen_height, args.screen_width, self.batch_size)
      self.s_t = self.be.empty(self.input_shape)
      self.s_t.lshape = self.input_shape # HACK: needed for convolutional networks
      self.target_q = self.be.empty((self.num_actions, self.batch_size))

      layers = self._create_neon_layers(num_actions)
      self.model = Model(layers = layers)
      self.loss = GeneralizedCost(costfunc = SumSquared())

      for l in self.model.layers.layers:
        l.parallelism = 'Disabled'

      self.model.initialize(self.input_shape[:-1], self.cost)
      self.optimizer = RMSProp(learning_rate = args.learning_rate,
          decay_rate = args.decay_rate,
          stochastic_round = stochastic_round)
    elif self.backend == 'tf':
      self.w = {}
      self.t_w = {}

      #initializer = tf.contrib.layers.xavier_initializer()
      initializer = tf.truncated_normal_initializer(0, 0.02)
      activation_fn = tf.nn.relu

      # training network
      if self.cnn_format == 'NCHW':
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_width, self.screen_height], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.screen_width, self.screen_height, self.history_length], name='s_t')

      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
      self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')
      self.q_action = tf.argmax(self.q, dimension=1)

      # target network
      if self.cnn_format == 'NCHW':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.history_length, self.screen_width, self.screen_height], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_width, self.screen_height, self.history_length], name='target_s_t')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
          linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
      self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
          linear(self.target_l4, self.env.action_size, name='target_q')

      # optimizer
      self.target_q_t = tf.placeholder('float32', [None])
      self.action = tf.placeholder('int64', [None])

      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0)
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1)

      self.delta = self.target_q_t - q_acted
      self.clipped_delta = tf.clip_by_value(self.delta, -1, 1)

      self.loss = tf.reduce_mean(tf.square(self.clipped_delta))
      self.optim = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.95, epsilon=0.01).minimize(self.loss)

      self.summary = tf.merge_all_summaries()
      self.writer = tf.train.SummaryWriter("./logs/%s" % self.model_dir, self.sess.graph)

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w[name].assign(self.w[name].eval()).eval()

  def _create_neon_layers(self):
    init_norm = Gaussian(loc=0.0, scale=0.01)
    layers = []

    layers.append(Conv((8, 8, 32), strides=4, init=init_norm, activation=Rectlin()))
    layers.append(Conv((4, 4, 64), strides=2, init=init_norm, activation=Rectlin()))
    layers.append(Conv((3, 3, 64), strides=1, init=init_norm, activation=Rectlin()))

    layers.append(Affine(nout=512, init=init_norm, activation=Rectlin()))
    layers.append(Affine(nout=self.env.action_size, init = init_norm))
    return layers
