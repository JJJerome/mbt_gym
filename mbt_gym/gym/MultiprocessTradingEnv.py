import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Iterable

import gym
import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _flatten_obs

STORE_TERMINAL_OBSERVATION_INFO = True


def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, infos = env.step(data)
                single_done = done[0] if len(done) > 1 else done
                if single_done:
                    if STORE_TERMINAL_OBSERVATION_INFO:
                        infos = infos.copy()
                        for count, info in enumerate(infos):
                            # save final observation where user can get it, then automatically reset (an SB3 convention).
                            info["terminal_observation"] = observation[count, :]
                    observation = env.reset()
                remote.send((observation, reward, done, infos))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class MultiprocessTradingEnv(SubprocVecEnv):
    """
    This is a slight modification of SubprocVecEnv, the details of which can be found at
    https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#subprocvecenv.

    In particular, it modifies it SubprocVecEnv that the inputs are already VecEnvs. This allows the user to choose the
    amount of vectorisation that is preformed via numpy (in VectorizedTradingEnvironment) and the amount of
    multiprocessing processes.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        super().__init__(self, env_fns, start_method)

        self.remotes[0].send(("get_attr", "num_trajectories"))
        num_trajectories_per_env = self.remotes[0].recv()

        self.remotes[0].send(("get_attr", "n_steps"))
        n_steps = self.remotes[0].recv()

        self.num_trajectories_per_env = num_trajectories_per_env
        self.num_multiprocess_envs = len(self.remotes)
        self.n_steps = n_steps
        self.num_trajectories = len(env_fns) * num_trajectories_per_env
        self.num_envs = self.num_trajectories

    def step_async(self, actions: np.ndarray) -> None:
        multi_actions = self.flatten_multi(actions, inverse=True)
        for remote, action in zip(self.remotes, multi_actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        obs = self.flatten_multi(_flatten_obs(obs, self.observation_space))
        rews = self.flatten_multi(np.stack(rews))
        dones = self.flatten_multi(np.stack(dones))
        return obs, rews, dones, list(np.stack(infos).reshape(-1))

    def flatten_multi(self, array: np.ndarray, inverse=False):
        if inverse:
            return list(array.reshape(self.num_multiprocess_envs, self.num_trajectories_per_env, -1))
        else:
            return array.reshape(self.num_multiprocess_envs * self.num_trajectories_per_env, -1).squeeze()

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_obs(obs, self.observation_space)
        return self.flatten_multi(obs)
