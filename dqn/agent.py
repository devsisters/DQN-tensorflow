import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .ops import linear, conv2d
from .replay_memory import ReplayMemory
from utils import get_time, save_pkl, load_pkl

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)
    self.sess = sess

    self.env = environment
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_dqn()

  def train(self):
    start_step = self.step_op.eval()
    start_time = time.time()

    num_game = 0
    total_reward = 0.
    self.total_loss = 0.
    self.total_q = 0.
    self.update_count = 0
    ep_reward = 0.
    ep_rewards = []

    action = 0
    warning_count = 0
    screen, reward, terminal = self.env.new_game()

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      self.history.add(screen)

      #if self.cnn_format == 'NCHW' and (self.history.get()[-1]-self.history.get()[-2]).sum() == 0.0 \
      #    or self.cnn_format == 'NHWC' and (self.history.get()[:,:,-1]-self.history.get().mean(2)).sum() == 0.0:
      #  warning_count += 1

      #  if warning_count > 30:
      #    import ipdb; ipdb.set_trace() 
      #else:
      #  warning_count = 0

      if self.step == self.learn_start:
        num_game = 0
        total_reward = 0.
        self.total_loss = 0.
        self.total_q = 0.
        self.update_count = 0
        ep_reward = 0.
        ep_rewards = []

      action = self.perceive(screen, reward, action, terminal)
      if terminal:
        screen, reward, terminal = self.env.new_game()
        num_game += 1

        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        screen, reward, terminal = self.env.act(action, is_training=True)
        ep_reward += reward

      total_reward += reward

      if self.step >= self.learn_start:
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
            self.inject_summary({
                'average/reward': avg_reward,
                'average/loss': avg_loss,
                'average/q': avg_q,
                'episode/max reward': max_ep_reward,
                'episode/min reward': min_ep_reward,
                'episode/min reward': avg_ep_reward,
                'episode/num of game': num_game,
                'episode/rewards': ep_rewards,
              }, self.step)

          num_game = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []

        if self.step % self.save_step == self.save_step - 1:
          self.step_assign_op.eval({self.step_input: self.step + 1})
          self.save_model(self.step + 1)

  def play(self, n_step=1000, n_episode=3, test_ep=0.01, render=False):
    test_history = History(self.config)

    if not self.display:
      self.env.env.monitor.start('/tmp/%s-%s' % (self.env_name, get_time()))

    for i_episode in xrange(n_episode):
      screen, reward, terminal = self.env.new_game()

      for _ in range(self.history_length):
        test_history.add(screen)

      for t in tqdm(range(n_step), ncols=70):
        if random.random() < test_ep:
          action = random.randint(0, self.env.action_size - 1)
        else:
          action = self.q_action.eval({self.s_t: [test_history.get()]})[0]

        screen, reward, terminal = self.env.act(action, is_training=False)
        test_history.add(screen)

        if terminal:
          break

    if not self.display:
      self.env.env.monitor.close()

  def perceive(self, screen, reward, action, terminal, test_ep=None):
    # reward clipping
    reward = max(self.min_reward, min(self.max_reward, reward))

    if test_ep == None:
      self.memory.add(screen, reward, action, terminal)

    # e greedy
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randint(0, self.env.action_size - 1)
    else:
      action = self.q_action.eval({self.s_t: [self.history.get()]})[0]

    if self.step > self.learn_start:
      if test_ep == None and self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

    return action

  def q_learning_mini_batch(self):
    if self.memory.count < self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

    #try:
    #  assert (s_t[0] - s_t[1]).mean() != 0.0
    #  assert len(np.unique(action)) != 1
    #except:
    #  import ipdb; ipdb.set_trace() 

    t = time.time()
    q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

    #try:
    #  assert q_t_plus_1.max() != 0.0
    #  assert q_t_plus_1[:,0].mean() != q_t_plus_1[0][0]
    #except:
    #  import ipdb; ipdb.set_trace() 

    terminal = np.array(terminal) + 0.
    max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
    target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    _, q_t, loss = self.sess.run([self.optim, self.q, self.loss], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
    })

    self.total_loss += loss
    self.total_q += q_t.mean()
    self.update_count += 1

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
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_width, self.screen_height, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.history_length, self.screen_width, self.screen_height], name='target_s_t')

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

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')

      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.target_q_t - q_acted
      self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

      self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
      self.optim = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.95, epsilon=0.01).minimize(self.loss)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average/reward', 'average/loss', 'average/q', \
          'episode/max reward', 'episode/min reward', 'episode/avg reward', 'episode/num of game']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary(tag, self.summary_placeholders[tag])

      histogram_summary_tags = ['episode/rewards']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

      self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=10)

    self.load_model()
    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def save_weight_to_pkl(self):
    for name in self.w.keys():
      save_pkl(self.w[name].eval(), "%s.pkl" % name)

  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl("%s.pkl" % name)})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)
