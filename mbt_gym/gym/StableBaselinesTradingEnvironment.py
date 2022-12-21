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
            self.linear_intercept_action = self.env.action_space.low
            self.linear_gradient_action = (self.env.action_space.high - self.env.action_space.low) / 2
            self.normalise_action: Callable = (
                lambda action: (action - self.linear_intercept_action) / self.linear_gradient_action - 1
            )
            self.denormalise_action: Callable = (
                lambda action: (action + 1) * self.linear_gradient_action + self.linear_intercept_action
            )
            action_space = gym.spaces.Box(
                low=-np.ones_like(self.env.action_space.low), high=np.ones_like(self.env.action_space.high)
            )
        else:
            action_space = self.env.action_space
        if self.normalise_observation_space:
            # Linear normalisation of the gym.Box space so that the domain of the observation space is [-1,1].
            self.linear_intercept_observation = self.env.observation_space.low
            self.linear_gradient_observation = (self.env.observation_space.high - self.env.observation_space.low) / 2
            self.normalise_observation: Callable = (
                lambda observation: (observation - self.linear_intercept_observation) / self.linear_gradient_observation
                - 1
            )
            self.denormalise_observation: Callable = (
                lambda observation: (observation + 1) * self.linear_gradient_observation
                + self.linear_intercept_observation
            )
            observation_space = gym.spaces.Box(
                low=-np.ones_like(self.env.observation_space.low), high=np.ones_like(self.env.observation_space.high)
            )
        else:
            observation_space = gym.spaces.Box(
                low=-np.ones_like(self.env.observation_space.low), high=np.ones_like(self.env.observation_space.high)
            )
        super().__init__(self.env.num_trajectories, observation_space, action_space)

    def reset(self) -> VecEnvObs:
        return self.env.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = self.denormalise_action(actions) if self.normalise_action_space else actions

    def step_wait(self) -> VecEnvStepReturn:
        state, rewards, dones, infos = self.env.step(self.actions)
        if self.normalise_observation_space:
            state = self.normalise_observation(state)
        if dones.min():
            if self.store_terminal_observation_info:
                for info, count in enumerate(infos):
                    # save final observation where user can get it, then automatically reset (an SB3 convention).
                    info["terminal_observation"] = state[:, count]
            state = self.env.reset()
        return state, rewards, dones, infos

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
