from typing import List, Any, Type, Optional, Union

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

from DRL4AMM.gym.MarketMakingEnvironment import MarketMakingEnvironment


class VectorizedMarketMakingEnvironmentSB(VecEnv):
    def __init__(self, market_making_env: MarketMakingEnvironment):
        self.env = market_making_env
        self.actions: np.ndarray = self.env.action_space.sample()
        super().__init__(self.env.num_trajectories, self.env._get_observation_space(), self.env._get_action_space())

    def reset(self) -> VecEnvObs:
        return self.env.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        return self.env.step(self.actions)

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
