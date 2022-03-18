from abc import abstractmethod

from atgym.atstrategies import AbstractStrategy

from stable_baselines3 import A2C, DQN, PPO


class RLStrategy(AbstractStrategy):
    """Abstract class for reinforcement learning-based strategies."""
    def __init__(self, obs_shape, name=None, env=None):
        if name is None:
            name = "rl_strategy"
        super().__init__(obs_shape=obs_shape, name=name)
        self.env = env
        self.model = None

    def predict(self, obs):
        return self.model.predict(obs)[0]

    def train(self, steps=20000):
        """Train agent on a trading environment. """
        self.model.learn(steps)


class A2CStrategy(RLStrategy):
    """Actor-critic algorithm combining policy and value-based methods."""
    def __init__(self, obs_shape, name=None, env=None):
        if name is None:
            name = "a2c_strategy"
        super().__init__(obs_shape=obs_shape, name=name, env=env)
        self.model = A2C("MlpPolicy", env, verbose=0)


class DQNStrategy(RLStrategy):
    """Deep Q-Network algorithm."""
    def __init__(self, obs_shape, name=None, env=None):
        if name is None:
            name = "dqn_strategy"
        super().__init__(obs_shape=obs_shape, name=name, env=env)
        self.model = DQN("MlpPolicy", env, verbose=0)


class PPOStrategy(RLStrategy):
    """Proximal Policy Optimization algorithm."""
    def __init__(self, obs_shape, name=None, env=None):
        if name is None:
            name = "ppo_strategy"
        super().__init__(obs_shape=obs_shape, name=name, env=env)
        self.model = PPO("MlpPolicy", env, verbose=0)
