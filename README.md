# Human-Level Control through Deep Reinforcement Learning

Tensorflow implementation of [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf).

![model](assets/model.png)

This implementation contains:

1. Epsilon-greedy policy and Deep Q-network
2. Experience replay memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learnig targets are fixed for intervals
    - to reduce the correlations between target and predicted Q-values


## Requirements

- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [OpenCV2](http://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)


## Usage

First, install prerequisites with:

    $ pip install tqdm gym[all]

To train a model for Breakout:

    $ python main.py --env_name=Breakout-v0 --is_train=True

To test a trained model for Breakout:

    $ python main.py --env_name=Breakout-v0 --is_train=True --display=True

Test and record with gym:

    $ python main.py --is_train=False


## Results

Result of training for 24 hours using GTX 980 ti.

![best](assets/best.gif)


## Training details

Details of `Breakout` with model `m2`(red) for 18 hours using GTX 980 Ti.

(`episode/min reward` should be `episode/average reward`. typo)

1. Statistics of loss, q values, rewards and # of game / episode
![tensorboard](assets/0518_scalar_step_m2.png)
![tensorboard](assets/0518_scalar_wall_m2.png)
2. Histogram of rewards / episode
![tensorboard](assets/0518_hist_step_m2.png)
![tensorboard](assets/0518_hist_wall_m2.png)

Details of `Breakout` with model `m1`(green), `m2`(purple), `m3`(blue) and `m4`(red) for 15 hours using GTX 980 Ti.

1. Statistics of loss, q values, rewards and # of game / episode
![tensorboard](assets/0519_scalar_step_all.png)
![tensorboard](assets/0519_scalar_wall_all.png)
2. Histogram of rewards / episode
![tensorboard](assets/0519_hist_step_all.png)
![tensorboard](assets/0519_hist_wall_all.png)


## References

- [simple_dqn](https://github.com/tambetm/simple_dqn.git)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)


## License

MIT License.
