import cv2
import gym
import random

class GymEnvironment(object):
  def __init__(self, env_name, screen_width, screen_height, action_repeat, random_start):
    self.env = gym.make(env_name)

    self._screen = None
    self.rewrad = 0
    self.terminal = True

    self.action_repeat = action_repeat
    self.random_start = random_start
    self.dims = (screen_width, screen_height)

  def new_game(self):
    self._screen = self.env.reset()

    while not self.terminal:
      self._random_step()

    self._step(0)
    return self.state

  def new_random_game(self):
    self.new_game()
    for _ in xrange(random.randint(0, self.random_start)):
      self._step(0)
    return self.screen, self.reward, 0, self.terminal

  def act(self, action, is_training):
    cumulated_reward = 0
    start_lives = self.lives

    for _ in xrange(self.action_repeat):
      self._step(action)
      cumulated_reward = cumulated_reward + self.reward

      if is_training and start_lives > self.lives:
        self.terminal = True
        break

    self.reward = cumulated_reward
    return self.state

  def _step(self, action):
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  def _random_step(self):
    action = x.action_space.sample()
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  @property
  def screen(self):
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
    self.obs = None
    self.terminal = None
    # OpenCV expects width as first and height as second s
    self.dims = (args.screen_width, args.screen_height)

  def numActions(self):
    assert isinstance(self.env.action_space, gym.spaces.Discrete)
    return self.env.action_space.n

  def new_random_game(self):
    self.obs = self.env.reset()
    self.terminal = False
    return self.getScreen(), 0, 0, self.terminal

  def act(self, action, is_training=True):
    self.obs, reward, self.terminal, _ = self.env.step(action)
    return self.getScreen(), reward, self.terminal

  def getScreen(self):
    assert self.obs is not None
    return cv2.resize(cv2.cvtColor(self.obs, cv2.COLOR_RGB2GRAY), self.dims)

  def isTerminal(self):
    assert self.terminal is not None
    return self.terminal

  @property
  def action_size(self):
    return self.env.action_space.n
