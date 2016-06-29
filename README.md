# Human-Level Control through Deep Reinforcement Learning

Tensorflow implementation of [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf).

![model](assets/model.png)

This implementation contains:

1. Deep Q-network and Q-learning
2. Experience replay memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals
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
    $ python main.py --env_name=Breakout-v0 --is_train=True --display=True

To test and record the screen with gym:

    $ python main.py --is_train=False
    $ python main.py --is_train=False --display=True


## Results

Result of training for 24 hours using GTX 980 ti.

![best](assets/best.gif)


## Simple Results

Details of `Breakout` with model `m2`(red) for 30 hours using GTX 980 Ti.

![tensorboard](assets/0620_scalar_step_m2.png)

Details of `Breakout` with model `m3`(red) for 30 hours using GTX 980 Ti.

![tensorboard](assets/0620_scalar_step_m3.png)


## Detailed Results

**[1] Action-repeat (frame-skip) of 1, 2, and 4 without learning rate decay**

![A1_A2_A4_0.00025lr](assets/A1_A2_A4_0.00025lr.png)

**[2] Action-repeat (frame-skip) of 1, 2, and 4 with learning rate decay**

![A1_A2_A4_0.0025lr](assets/A1_A2_A4_0.0025lr.png)

**[1] & [2]**

![A1_A2_A4_0.00025lr_0.0025lr](assets/A1_A2_A4_0.00025lr_0.0025lr.png)


**[3] Action-repeat of 4 for DQN (dark blue) Dueling DQN (dark green) DDQN (brown) Dueling DDQN (turquoise)**

The current hyper parameters and gradient clipping are not implemented as it is in the paper.

![A4_duel_double](assets/A4_duel_double.png)


**[4] Distributed action-repeat (frame-skip) of 4 without learning rate decay**

![A4_0.00025lr_distributed](assets/A4_0.00025lr_distributed.png)


## References

- [simple_dqn](https://github.com/tambetm/simple_dqn.git)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)


## License

MIT License.
