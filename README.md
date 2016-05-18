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

Trained in GPU but test in CPU with GPU weights:

    $ python main.py --model=m2 --save=True # save pickle from checkpoints
    $ python main.py --model=m2 --load=True --cpu=True --display=True --is_train=False


## Results

Training details of Breakout trained with model `m1` (blue) and `m2` (red) for 10 hours with GTX 980 Ti.

(`episode/min reward` should be `episode/average reward`. typo)

![tensorboard](assets/tensorboard_160518_scalar1.png)

![tensorboard](assets/tensorboard_160518_histogram1.png)

Training details of Breakout trained with model `m3` (blue) and `m4` (red) for 10 hours with GTX 980 Ti.

(`episode/min reward` should be `episode/average reward`. typo)

![tensorboard](assets/tensorboard_160518_scalar2.png)

![tensorboard](assets/tensorboard_160518_histogram2.png)



## References

- [async_rl](https://github.com/muupan/async-rl)
- [simple_dqn](https://github.com/tambetm/simple_dqn.git)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)


## License

MIT License.
