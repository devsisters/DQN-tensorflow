import cv2
import gym
import random

class GymEnvironment(object):
  def __init__(self, env_name, screen_width, screen_height, action_repeat, random_start):
    self.env = gym.make(env_name)

    # raw screen without normalization
    self._screen = None
    self.rewrad = 0
    self.terminal = True

    self.action_repeat = action_repeat
    self.random_start = random_start
    self.dims = (screen_width, screen_height)

  def new_game(self):
    self._screen = self.env.reset()
    self._step(0)
    return self.screen, 0, self.terminal

  def new_random_game(self):
    self.new_game()
    for _ in xrange(random.randint(0, self.random_start)):
      self._step(0)
    return self.screen, 0, self.terminal

  def act(self, action, is_training):
    cumulated_reward = 0
    start_lives = self.lives

    for _ in xrange(self.action_repeat):
      self._step(action)
      cumulated_reward = cumulated_reward + self.reward

      if is_training and start_lives > self.lives:
        self.terminal = True
        break

      if self.terminal:
        break

    self.reward = cumulated_reward
    return self.state

  def _step(self, action):
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  def _random_step(self):
    action = self.env.action_space.sample()
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  @property
  def screen(self): # normalized screen
    return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY)/255., self.dims)
    #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

  @property
  def action_size(self):
    return self.env.action_space.n

  @property
  def lives(self):
    return self.env.ale.lives()

class SimpleGymEnvironment(object):
  # For use with Open AI Gym Environment
  def __init__(self, env_id, args):
    self.env = gym.make(env_id)

    self._screen = None
    self.rewrad = 0
    self.terminal = None

    self.action_repeat = action_repeat
    self.random_start = random_start

    self.dims = (args.screen_width, args.screen_height)

  def new_game(self):
    self._screen = self.env.reset()
    self._step(0)
    return self.screen, 0, self.terminal

  def new_random_game(self):
    self.new_game()
    for _ in xrange(random.randint(0, self.random_start)):
      self.act(0)
    return self.screen, 0, self.terminal

  def numActions(self):
    assert isinstance(self.env.action_space, gym.spaces.Discrete)
    return self.env.action_space.n

  def act(self, action, is_training=True):
    self.env._step(action)
    return self.screen, self.reward, terminal

  def _step(self, action):
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  def _random_step(self):
    action = self.env.action_space.sample()
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  @ property
  def screen(self):
    assert self._screen is not None
    return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY)/255., self.dims)

  @property
  def action_size(self):
    return self.env.action_space.n
