"""Code modification of https://github.com/tambetm/simple_dqn"""

import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Gaussian
from neon.optimizers import RMSProp, Adam, Adadelta
from neon.layers import Affine, Conv, GeneralizedCost
from neon.transforms import Rectlin
from neon.models import Model
from neon.transforms import SumSquared
from neon.util.persist import save_obj

from .base import BaseModel
from .history import History
from .ops import linear, conv2d
from .replay_memory import ReplayMemory
from utils import get_time, save_pkl, load_pkl

class Agent(BaseModel):
  def __init__(self, config, environment, sess, random_seed):
    super(Agent, self).__init__(config)
    self.sess = sess

    self.random_seed = random_seed
    self.datatype = 'float32'
    self.batch_norm = False
    self.stochastic_round = False
    self.backend = 'gpu'
    self.screen_dim = (self.screen_height, self.screen_width)
    self.num_actions = environment.action_size
    self.save_weights_prefix = "neon_net"

    self.env = environment
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_dqn()

  def act(self, exploration_rate):
    # exploration rate determines the probability of random moves
    if random.random() < exploration_rate:
      action = random.randrange(self.num_actions)
    else:
      # otherwise choose action with highest Q-value
      state = self.history.get()
      # for convenience getStateMinibatch() returns minibatch
      # where first item is the current state
      qvalues = self.predict(state)
      assert len(qvalues[0]) == self.num_actions
      # choose highest Q-value of first state
      action = np.argmax(qvalues[0])

    # perform the action
    screen, reward, terminal = self.env.act(action)

    # add screen to buffer
    self.history.add(screen)

    # restart the game if over
    if terminal:
      self.env.new_random_game()

    # call callback to record statistics
    if self.callback:
      self.callback.on_step(action, reward, terminal, screen, exploration_rate)

    return action, reward, screen, terminal

  def train(self):
    start_step = self.step_op.eval()
    start_time = time.time()

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    ep_rewards, actions = [], []

    screen, reward, action, terminal = self.env.new_random_game()

    for _ in range(self.history_length):
      self.history.add(screen)

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      if self.step == self.learn_start:
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []

      # 1. perform game step
      action, reward, screen, terminal = self.act(self._explorationRate())
      # 2. observe
      self.observe(screen, reward, action, terminal)

      self.memory.add(screen, reward, action, terminal)
      if self.memory.count > self.memory.batch_size and self.step % self.train_frequency == 0:
        # train the network
        self.q_learning_mini_batch()

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        ep_reward += reward

      actions.append(action)
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
                'episode/avg reward': avg_ep_reward,
                'episode/num of game': num_game,
                'episode/rewards': ep_rewards,
                'episode/actions': actions,
              }, self.step)

          num_game = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actions = []

        if self.step % self.save_step == self.save_step - 1:
          self.step_assign_op.eval({self.step_input: self.step + 1})
          self.save_model(self.step + 1)

  def predict(self, s_t, test_ep=None):
    # minibatch is full size, because Neon doesn't let change the minibatch size
    assert states.shape == ((self.batch_size, self.history_length,) + self.screen_dim)

    # calculate Q-values for the states
    self._setInput(states)
    qvalues = self.model.fprop(self.input, inference = True)
    assert qvalues.shape == (self.num_actions, self.batch_size)

    # transpose the result, so that batch size is first dimension
    return qvalues.T.asnumpyarray()

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

  def q_learning_mini_batch(self):
    if self.memory.count < self.history_length:
      return
    else:
      prestates, actions, rewards, poststates, terminals = self.memory.sample()

    # expand components of minibatch
    assert len(prestates.shape) == 4
    assert len(poststates.shape) == 4
    assert len(actions.shape) == 1
    assert len(rewards.shape) == 1
    assert len(terminals.shape) == 1
    assert prestates.shape == poststates.shape
    assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

    if self.target_q_update_step and self.train_iterations % self.target_q_update_step == 0:
      # have to serialize also states for batch normalization to work
      pdict = self.model.get_description(get_weights=True, keep_states=True)
      self.target_model.deserialize(pdict, load_states=True)

    # feed-forward pass for poststates to get Q-values
    self._setInput(poststates)
    postq = self.target_model.fprop(self.input, inference = True)
    assert postq.shape == (self.num_actions, self.batch_size)

    # calculate max Q-value for each poststate
    maxpostq = self.be.max(postq, axis=0).asnumpyarray()
    assert maxpostq.shape == (1, self.batch_size)

    # feed-forward pass for prestates
    self._setInput(prestates)
    preq = self.model.fprop(self.input, inference = False)
    assert preq.shape == (self.num_actions, self.batch_size)

    # make copy of prestate Q-values as targets
    targets = preq.asnumpyarray()

    # clip rewards between -1 and 1
    rewards = np.clip(rewards, self.min_reward, self.max_reward)

    # update Q-value targets for actions taken
    for i, action in enumerate(actions):
      if terminals[i]:
        targets[action, i] = float(rewards[i])
      else:
        targets[action, i] = float(rewards[i]) + self.discount_rate * maxpostq[0,i]

    # copy targets to GPU memory
    self.targets.set(targets)

    # calculate errors
    deltas = self.cost.get_errors(preq, self.targets)
    assert deltas.shape == (self.num_actions, self.batch_size)
    #assert np.count_nonzero(deltas.asnumpyarray()) == 32

    # calculate cost, just in case
    cost = self.cost.get_cost(preq, self.targets)
    assert cost.shape == (1,1)

    # clip errors
    if self.clip_error:
      self.be.clip(deltas, -self.clip_error, self.clip_error, out = deltas)

    # perform back-propagation of gradients
    self.model.bprop(deltas)

    # perform optimization
    self.optimizer.optimize(self.model.layers_to_optimize, epoch)

    # increase number of weight updates (needed for target clone interval)
    self.train_iterations += 1

    # calculate statistics
    if self.callback:
      self.callback.on_train(cost.asnumpyarray()[0,0])

    self.writer.add_summary(summary_str, self.step)
    self.total_loss += loss
    self.total_q += q_t.mean()
    self.update_count += 1

  def build_dqn(self):
    self.be = gen_backend(backend = self.backend,
                 batch_size = self.batch_size,
                 rng_seed = self.random_seed,
                 datatype = np.dtype(self.datatype).type,
                 stochastic_round = self.stochastic_round)

    # prepare tensors once and reuse them
    self.input_shape = (self.history_length,) + self.screen_dim + (self.batch_size,)
    self.input = self.be.empty(self.input_shape)
    self.input.lshape = self.input_shape # HACK: needed for convolutional networks
    self.targets = self.be.empty((self.num_actions, self.batch_size))

    # create model
    layers = self._createLayers(self.num_actions)
    self.model = Model(layers = layers)
    self.cost = GeneralizedCost(costfunc = SumSquared())

    # Bug fix
    for l in self.model.layers.layers:
      l.parallelism = 'Disabled'

    import ipdb; ipdb.set_trace() 

    self.optimizer = RMSProp(learning_rate = self.learning_rate, 
        decay_rate = self.discount, 
        stochastic_round = self.stochastic_round)

    # create target model
    self.target_q_update_step = self.target_q_update_step
    self.train_iterations = 0
    if self.target_q_update_step:
      self.target_model = Model(layers = self._createLayers(self.num_actions))

      # Bug fix
      for l in self.target_model.layers.layers:
        l.parallelism = 'Disabled'

      self.target_model.initialize(self.input_shape[:-1])
      self.save_weights_prefix = self.save_weights_prefix
    else:
      self.target_model = self.model

    self.callback = None

    tf.initialize_all_variables().run()
    self._saver = tf.train.Saver([self.step_op], max_to_keep=10)
    self.load_model()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def play(self, n_step=10000, n_episode=1000, test_ep=None, render=False):
    if test_ep == None:
      #test_ep = self.ep_end
      test_ep = 1e-100

    test_history = History(self.config)

    if not self.display:
      self.env.env.monitor.start('/tmp/%s-%s' % (self.env_name, get_time()))

    for i_episode in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()

      for _ in range(self.history_length):
        test_history.add(screen)

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(test_history.get(), test_ep)
        # 2. act
        screen, reward, terminal = self.env.act(action, is_training=False)
        # 3. observe
        test_history.add(screen)

        if terminal:
          break

    if not self.display:
      self.env.env.monitor.close()

  def load_weights(self, load_path):
    self.model.load_params(load_path)

  def save_weights(self, save_path):
    self.model.save_params(save_path)

  def _createLayers(self, num_actions):
    # create network
    init_norm = Gaussian(loc=0.0, scale=0.01)
    layers = []
    # The first hidden layer convolves 32 filters of 8x8 with stride 4 with the input image and applies a rectifier nonlinearity.
    layers.append(Conv((8, 8, 32), strides=4, init=init_norm, activation=Rectlin(), batch_norm=self.batch_norm))
    # The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
    layers.append(Conv((4, 4, 64), strides=2, init=init_norm, activation=Rectlin(), batch_norm=self.batch_norm))
    # This is followed by a third convolutional layer that convolves 64 filters of 3x3 with stride 1 followed by a rectifier.
    layers.append(Conv((3, 3, 64), strides=1, init=init_norm, activation=Rectlin(), batch_norm=self.batch_norm))
    # The final hidden layer is fully-connected and consists of 512 rectifier units.
    layers.append(Affine(nout=512, init=init_norm, activation=Rectlin(), batch_norm=self.batch_norm))
    # The output layer is a fully-connected linear layer with a single output for each valid action.
    layers.append(Affine(nout=num_actions, init = init_norm))
    return layers

  def _setInput(self, states):
    # change order of axes to match what Neon expects
    states = np.transpose(states, axes = (1, 2, 3, 0))
    # copy() shouldn't be necessary here, but Neon doesn't work otherwise
    self.input.set(states.copy())
    # normalize network input between 0 and 1
    self.be.divide(self.input, 255, self.input)

  def _restartRandom(self):
    self.env.restart()
    # perform random number of dummy actions to produce more stochastic games
    for i in xrange(random.randint(self.history_length, self.random_starts) + 1):
      screen, reward, terminal = self.env.act(0)
      assert not terminal, "terminal state occurred during random initialization"
      # add dummy states to buffer
      self.history.add(screen)

  def _explorationRate(self):
    # calculate decaying exploration rate
    if self.step < self.ep_end_t:
      return self.ep_start - self.step * (self.ep_start - self.ep_end) / self.ep_end_t
    else:
      return self.ep_end
