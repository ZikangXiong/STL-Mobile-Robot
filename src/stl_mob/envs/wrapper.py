from abc import ABC, abstractmethod
from typing import Union, List

import gym
import numpy as np

from stl_mob.stl.tasks import TaskBase


class EnvWrapper(ABC):
    def __init__(self, task: TaskBase):
        self.task = task
        self.obstacle_list = self.task.task_map.obs_list
        self.wp_list = self.task.wp_list
        self._goal = None
        self.env = self.build_env()

    @abstractmethod
    def _set_goal(self):
        raise NotImplementedError()

    @abstractmethod
    def build_env(self) -> gym.Env:
        raise NotImplementedError()

    def set_goal(self, goal: Union[List, np.ndarray]):
        self._set_goal()
        self._goal = np.array(goal)

    def get_goal(self):
        return self._goal

    def step(self, action: Union[List, np.ndarray]):
        return self.env.step(action)


class MujocoEnv(EnvWrapper, ABC):
    def __init__(self, task: TaskBase):
        super(MujocoEnv, self).__init__(task)


class PybulletEnv(EnvWrapper, ABC):
    def __init__(self, task: TaskBase):
        super(PybulletEnv, self).__init__(task)


class PointEnv(EnvWrapper):
    def __init__(self, task: TaskBase):
        super(PointEnv, self).__init__(task)


class CarEnv(EnvWrapper):
    def __init__(self, task: TaskBase):
        super(CarEnv, self).__init__(task)


class DoggoEnv(EnvWrapper):
    def __init__(self, task: TaskBase):
        super(DoggoEnv, self).__init__(task)


class DroneEnv(EnvWrapper):
    def __init__(self, task: TaskBase):
        super(DroneEnv, self).__init__(task)


def get_envs(env_name: str, task: TaskBase):
    if env_name == "drone":
        return DroneEnv(task)
    elif env_name == "point":
        return PointEnv(task)
    elif env_name == "car":
        return CarEnv(task)
    elif env_name == "doggo":
        return DoggoEnv(task)
    else:
        raise ValueError(f"Env {env_name} not found")
