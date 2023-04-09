from abc import ABC, abstractmethod
from typing import Union, List

import gym
import numpy as np

from stl_mob.envs.mujoco_robots.robots.engine import Engine
from stl_mob.envs.pybullet_robots.base import BulletEnv
from stl_mob.envs.pybullet_robots.robots.drone import Drone
from stl_mob.stl.tasks import TaskBase


class EnvWrapper(ABC):
    def __init__(self, task: TaskBase, enable_gui: bool = True):
        self.task = task
        self.enable_gui = enable_gui
        self.obstacle_list = self.task.task_map.obs_list
        self.wp_list = self.task.wp_list
        self._goal = None
        self.gym_env = self.build_env()

    @abstractmethod
    def _set_goal(self, goal: Union[List, np.ndarray]):
        raise NotImplementedError()

    @abstractmethod
    def build_env(self) -> gym.Env:
        raise NotImplementedError()

    def set_goal(self, goal: Union[List, np.ndarray]):
        self._set_goal(goal)
        self._goal = np.array(goal)

    def get_goal(self):
        return self._goal

    def step(self, action: Union[List, np.ndarray]):
        if self.enable_gui:
            self.gym_env.render()
        return self.gym_env.step(action)

    def reset(self):
        return self.gym_env.reset()


class MujocoEnv(EnvWrapper, ABC):
    def __init__(self, task: TaskBase):
        super(MujocoEnv, self).__init__(task)

    def _set_goal(self, goal: Union[List, np.ndarray]):
        pass


class PybulletEnv(EnvWrapper, ABC):
    def __init__(self, task: TaskBase):
        super(PybulletEnv, self).__init__(task)

    def _set_goal(self, goal: Union[List, np.ndarray]):
        pass


class PointEnv(MujocoEnv):
    def __init__(self, task: TaskBase):
        super(PointEnv, self).__init__(task)

    def build_env(self) -> gym.Env:
        return Engine({"robot_base": "xmls/point.xml"})


class CarEnv(MujocoEnv):
    def __init__(self, task: TaskBase):
        super(CarEnv, self).__init__(task)

    def build_env(self) -> gym.Env:
        return Engine({"robot_base": "xmls/car.xml"})


class DoggoEnv(MujocoEnv):
    def __init__(self, task: TaskBase):
        super(DoggoEnv, self).__init__(task)

    def build_env(self) -> gym.Env:
        return Engine({"robot_base": "xmls/doggo.xml"})


class DroneEnv(PybulletEnv):
    def __init__(self, task: TaskBase):
        super(DroneEnv, self).__init__(task)

    def build_env(self) -> gym.Env:
        return BulletEnv(Drone(enable_gui=self.enable_gui))


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
