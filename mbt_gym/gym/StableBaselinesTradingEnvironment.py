from typing import List, Any, Type, Optional, Union

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

from mbt_gym.gym.TradingEnvironment import TradingEnvironment


class StableBaselinesTradingEnvironment(VecEnv):
    def __init__(self, trading_env: TradingEnvironment, store_terminal_observation_info: bool = True):
        self.env = trading_env
        self.store_terminal_observation_info = store_terminal_observation_info
        self.actions: np.ndarray = self.env.action_space.sample()
        super().__init__(self.env.num_trajectories, self.env.observation_space, self.env.action_space)

    def reset(self) -> VecEnvObs:
        return self.env.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        state, rewards, dones, infos = self.env.step(self.actions)
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
        self.env.seed(seed)
