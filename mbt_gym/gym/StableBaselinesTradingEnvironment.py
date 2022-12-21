from copy import deepcopy
from typing import List, Any, Type, Optional, Union, Callable, Sequence

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

from mbt_gym.gym.TradingEnvironment import TradingEnvironment


class StableBaselinesTradingEnvironment(VecEnv):
    def __init__(
        self,
        trading_env: TradingEnvironment,
        store_terminal_observation_info: bool = True,
        normalise_action_space: bool = True,
        normalise_observation_space: bool = True,
    ):
        self.env = trading_env
        self.store_terminal_observation_info = store_terminal_observation_info
        self.normalise_action_space = normalise_action_space
        self.normalise_observation_space = normalise_observation_space
        self.actions: np.ndarray = self.env.action_space.sample()
        if self.normalise_action_space:
            # We just do a linear normalisation of the gym.Box space so that the domain of the action space is [-1,1].
            action_space = gym.spaces.Box(
                low=-np.ones_like(self.env.action_space.low), high=np.ones_like(self.env.action_space.high),
                dtype=np.float32
            )
        else:
            action_space = self.env.action_space
        if self.normalise_observation_space:
            # Linear normalisation of the gym.Box space so that the domain of the observation space is [-1,1].
            observation_space = gym.spaces.Box(
                low=-np.ones_like(self.env.observation_space.low), high=np.ones_like(self.env.observation_space.high),
                dtype=np.float32
            )
        else:
            observation_space = self.env.observation_space
        super().__init__(self.env.num_trajectories, observation_space, action_space)

    def reset(self) -> VecEnvObs:
        return self.normalise_observation(self.env.reset())

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = self.normalise_action(actions, inverse=True)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.env.step(self.actions)
        obs = self.normalise_observation(obs)
        if dones.min():
            if self.store_terminal_observation_info:
                infos = infos.copy()
                for count, info in enumerate(infos):
                    # save final observation where user can get it, then automatically reset (an SB3 convention).
                    info["terminal_observation"] = obs[count, :]
            obs = self.normalise_observation(self.env.reset())
        return obs, rewards, dones, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in range(self.env.num_trajectories)]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.env.seed(seed)

    def get_images(self) -> Sequence[np.ndarray]:
        pass

    def normalise_observation(self, obs: np.ndarray, inverse: bool = False):
        if self.normalise_observation_space and not inverse:
            return (obs - self.linear_intercept_obs) / self.linear_gradient_obs - 1
        elif self.normalise_observation_space and inverse:
            return (obs + 1) * self.linear_gradient_obs + self.linear_intercept_obs
        else:
            return obs

    def normalise_action(self, action: np.ndarray, inverse: bool = False):
        if self.normalise_action_space and not inverse:
            return (action - self.linear_intercept_action) / self.linear_gradient_action - 1
        elif self.normalise_action_space and inverse:
            return (action + 1) * self.linear_gradient_action + self.linear_intercept_action
        else:
            return action

    @property
    def linear_intercept_obs(self):
        return self.env.observation_space.low

    @property
    def linear_gradient_obs(self):
        return (self.env.observation_space.high - self.env.observation_space.low) / 2

    @property
    def linear_intercept_action(self):
        return self.env.action_space.low

    @property
    def linear_gradient_action(self):
        return (self.env.action_space.high - self.env.action_space.low) / 2
