
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .ops import linear, conv2d
from utils import get_time

class Agent(BaseModel):
  def __init__(self, config, environment, optimizer, lr_op):
    super(Agent, self).__init__(config)
    self.weight_dir = 'weights'

    self.env = environment
    self.history = History(self.config)

    self.lr_op = lr_op
    self.optimizer = optimizer

    self.step_op = tf.Variable(0, trainable=False, name='step')
    self.step_inc_op = self.step_op.assign_add(1, use_locking=True)
    self.build_dqn()

    self.saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)
    self.summary_op = tf.merge_all_summaries()
    self.init_op = tf.initialize_all_variables()

  def before_train(self, is_chief):
    self.T = self.step = self.step_op.eval(session=self.sess)
    screen, reward, action, terminal = self.env.new_random_game()

    for _ in xrange(self.history_length):
      self.history.add(screen)

    self.batch_s_t = [self.history.copy()]
    self.batch_reward = []
    self.batch_action = []
    self.batch_terminal = []

    if is_chief:
      iterator = tqdm(xrange(self.step, self.max_step), ncols=70, initial=self.step)
    else:
      iterator = xrange(self.step, self.max_step)

    return screen, reward, action, terminal, iterator

  def train(self, sv, is_chief):
    screen, reward, action, terminal, iterator = self.before_train(is_chief)

    for self.step in iterator:
      if self.step >= self.max_step:
        sv.request_stop()

      # 1. predict
      action = self.predict(self.history.get())
      # 2. act
      screen, reward, terminal = self.env.act(action, is_training=True)
      # 3. observe
      self.observe(screen, reward, action, terminal)

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()

  def train_with_summary(self, sv, is_chief):
    screen, reward, action, terminal, iterator = self.before_train(is_chief)

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    ep_rewards, actions = [], []

    for self.step in iterator:
      if self.step >= self.max_step:
        sv.request_stop()

      if self.step == self.learn_start:
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []

      # 1. predict
      action = self.predict(self.history.get())
      # 2. act
      screen, reward, terminal = self.env.act(action, is_training=True)
      # 3. observe
      self.observe(screen, reward, action, terminal, is_chief=True)

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        ep_reward += reward

      actions.append(action)
      total_reward += reward

      if self.step % self.test_step == self.test_step - 1:
        avg_reward = total_reward / self.test_step
        avg_loss = self.total_loss / self.update_count
        avg_q = self.total_q / self.update_count

        try:
          max_ep_reward = np.max(ep_rewards)
          min_ep_reward = np.min(ep_rewards)
          avg_ep_reward = np.mean(ep_rewards)
        except:
          max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

        print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
            % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

        if self.step > 180:
          self.inject_summary(sv, {
              'average.reward': avg_reward,
              'average.loss': avg_loss,
              'average.q': avg_q,
              'episode.max reward': max_ep_reward,
              'episode.min reward': min_ep_reward,
              'episode.avg reward': avg_ep_reward,
              'episode.num of game': num_game,
              'episode.rewards': ep_rewards,
              'episode.actions': actions,
              'training.learning_rate': self.lr,
            }, self.T)

        num_game = 0
        total_reward = 0.
        self.total_loss = 0.
        self.total_q = 0.
        self.update_count = 0
        ep_rewards = []
        actions = []

  def predict(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.q_action.eval({self.s_t: [s_t]}, session=self.sess)[0]

    return action

  def observe(self, screen, reward, action, terminal, is_chief=False):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.batch_s_t.append(self.history.copy())
    self.batch_action.append(action)
    self.batch_reward.append(reward)
    self.batch_terminal.append(terminal)

    if self.step % self.train_frequency == 0:
      self.batch_update(is_chief)

    self.T = self.sess.run(self.step_inc_op)
    if self.T % self.target_q_update_step == self.target_q_update_step - 1:
      self.update_target_q_network()

  def batch_update(self, is_chief):
    s_t, action, reward, s_t_plus_1, terminal = \
        self.batch_s_t[:-1], self.batch_action, self.batch_reward, self.batch_s_t[1:], self.batch_terminal

    #assert len(s_t) == self.batch_size and len(action) == self.batch_size
    #assert np.array_equal(s_t[0][1:], s_t_plus_1[0][:-1])

    if self.double_q:
      # Double Q-learning
      pred_action = self.q_action.eval({self.s_t: s_t_plus_1}, session=self.sess)

      q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
        self.target_s_t: s_t_plus_1,
        self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
      }, session=self.sess)
      target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
    else:
      q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1}, session=self.sess)

      terminal = np.array(terminal) + 0.
      max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
      target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    _, q_t, loss = self.sess.run([self.optim, self.q, self.loss], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.lr_op: self.lr,
    })

    if is_chief:
      self.total_loss += loss
      self.total_q += q_t.mean()
      self.update_count += 1

    self.batch_s_t = [self.history.copy()]
    self.batch_reward = []
    self.batch_action = []
    self.batch_terminal = []

  def build_dqn(self):
    self.w = {}
    self.t_w = {}

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # training network
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_width, self.screen_height, self.history_length], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_width, self.screen_height], name='s_t')

      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t/255.,
          16, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          32, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')

      shape = self.l2.get_shape().as_list()
      self.l2_flat = tf.reshape(self.l2, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.value_hid, self.w['l3_val_w'], self.w['l3_val_b'] = \
            linear(self.l2_flat, 256, activation_fn=activation_fn, name='value_hid')

        self.adv_hid, self.w['l3_adv_w'], self.w['l3_adv_b'] = \
            linear(self.l2_flat, 256, activation_fn=activation_fn, name='adv_hid')

        self.value, self.w['val_w_out'], self.w['val_w_b'] = \
          linear(self.value_hid, 1, name='value_out')

        self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
          linear(self.adv_hid, self.env.action_size, name='adv_out')

        # Average Dueling
        self.q = self.value + (self.advantage - 
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        self.l3, self.w['l3_w'], self.w['l3_b'] = linear(self.l2_flat, 256, activation_fn=activation_fn, name='l3')
        self.q, self.w['q_w'], self.w['q_b'] = linear(self.l3, self.env.action_size, name='q')

      self.q_action = tf.argmax(self.q, dimension=1)

    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_width, self.screen_height, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.history_length, self.screen_width, self.screen_height], name='target_s_t')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t/255., 
          16, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
          32, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')

      shape = self.target_l2.get_shape().as_list()
      self.target_l2_flat = tf.reshape(self.target_l2, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.t_value_hid, self.t_w['l3_val_w'], self.t_w['l3_val_b'] = \
            linear(self.target_l2_flat, 256, activation_fn=activation_fn, name='target_value_hid')

        self.t_adv_hid, self.t_w['l3_adv_w'], self.t_w['l3_adv_b'] = \
            linear(self.target_l2_flat, 256, activation_fn=activation_fn, name='target_adv_hid')

        self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
          linear(self.t_value_hid, 1, name='target_value_out')

        self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
          linear(self.t_adv_hid, self.env.action_size, name='target_adv_out')

        # Average Dueling
        self.target_q = self.t_value + (self.t_advantage - 
          tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
      else:
        self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = \
            linear(self.target_l2_flat, 256, activation_fn=activation_fn, name='target_l3')
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
            linear(self.target_l3, self.env.action_size, name='target_q')

      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_assign_op[name] = self.t_w[name].assign(self.w[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')

      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.target_q_t - q_acted
      self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')

      new_grads_and_vars = []
      grads_and_vars = self.optimizer.compute_gradients(self.loss, self.w.values())
      for grad, var in tuple(grads_and_vars):
        new_grads_and_vars.append((tf.clip_by_norm(grad, 40), var))

      self.optim = self.optimizer.apply_gradients(new_grads_and_vars)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      self.summary_op = tf.merge_summary(self.summary_ops.values(), name='total_summary')

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval(session=self.sess)

  def inject_summary(self, sv, tag_dict, step):
    sv.summary_computed(self.sess, self.sess.run(self.summary_op, {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    }))

  def play(self, sv, is_chief, n_step=10000, n_episode=100, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    if not self.display:
      gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
      self.env.env.monitor.start(gym_dir)

    best_reward, best_idx = 0, 0
    for idx in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()
      current_reward = 0

      for _ in xrange(self.history_length):
        test_history.add(screen)

      for t in tqdm(xrange(n_step), ncols=70):
        # 1. predict
        action = self.predict(test_history.get(), test_ep)
        # 2. act
        screen, reward, terminal = self.env.act(action, is_training=False)
        # 3. observe
        test_history.add(screen)

        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print "="*30
      print " [%d] Best reward : %d" % (best_idx, best_reward)
      print "="*30

    if not self.display:
      self.env.env.monitor.close()
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')

  @property
  def lr(self):
    return (self.max_step - self.step + 1.) / self.max_step * self.learning_rate

