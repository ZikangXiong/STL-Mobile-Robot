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
        self.gym_env: Union[Engine, BulletEnv] = self.build_env()

    @abstractmethod
    def _set_goal(self, goal: Union[List, np.ndarray]):
        raise NotImplementedError()

    @abstractmethod
    def build_env(self) -> Union[Engine, BulletEnv]:
        raise NotImplementedError()

    @abstractmethod
    def get_pos(self):
        pass

    def set_goal(self, goal: Union[List, np.ndarray]):
        self._set_goal(goal)
        self._goal = np.array(goal)

    def get_goal(self) -> np.ndarray:
        return self._goal

    def step(self, action: Union[List, np.ndarray]):
        if self.enable_gui:
            self.gym_env.render()
        return self.gym_env.step(action)

    def reset(self):
        return self.gym_env.reset()

    def reached(self, reach_radius: float = 0.3) -> bool:
        return np.linalg.norm(self.get_pos() - self.get_goal()) < reach_radius


class MujocoEnv(EnvWrapper, ABC):
    metadata = {"render.modes": ["human", "rgb_array"]}

    BASE_SENSORS = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']

    def __init__(self, task: TaskBase, enable_gui: bool = True):
        super(MujocoEnv, self).__init__(task, enable_gui)

    def _set_goal(self, goal: Union[List, np.ndarray]):
        self.gym_env.set_goal_position(goal_xy=goal[:2])

    def get_pos(self) -> np.ndarray:
        return np.array(self.gym_env.robot_pos[:2])


class PybulletEnv(EnvWrapper, ABC):
    def __init__(self, task: TaskBase, enable_gui: bool = True):
        super(PybulletEnv, self).__init__(task, enable_gui)

    def _set_goal(self, goal: Union[List, np.ndarray]):
        pass

    def get_pos(self) -> np.ndarray:
        # np.array(p.getBasePositionAndOrientation(self.robot_id, self.client_id)[0])
        pass


class PointEnv(MujocoEnv):

    def build_env(self) -> gym.Env:
        config = {
            "robot_base": f"xmls/point.xml",
            'sensors_obs': self.BASE_SENSORS,
            'observe_com': False,
            'observe_goal_comp': True
        }
        return Engine(config)


class CarEnv(MujocoEnv):

    def build_env(self) -> gym.Env:
        config = {
            "robot_base": f"xmls/car.xml",
            'sensors_obs': self.BASE_SENSORS,
            'observe_com': False,
            'observe_goal_comp': True,
            'box_size': 0.125,  # Box half-radius size
            'box_keepout': 0.125,  # Box keepout radius for placement
            'box_density': 0.0005
        }
        return Engine(config)


class DoggoEnv(MujocoEnv):

    def build_env(self) -> gym.Env:
        extra_sensor = [
            'touch_ankle_1a',
            'touch_ankle_2a',
            'touch_ankle_3a',
            'touch_ankle_4a',
            'touch_ankle_1b',
            'touch_ankle_2b',
            'touch_ankle_3b',
            'touch_ankle_4b'
        ]
        config = {
            "robot_base": f"xmls/doggo.xml",
            'sensors_obs': self.BASE_SENSORS + extra_sensor,
            'observe_com': False,
            'observe_goal_comp': True
        }
        return Engine(config)


class DroneEnv(PybulletEnv):

    def build_env(self) -> gym.Env:
        return BulletEnv(Drone(enable_gui=self.enable_gui))


def get_env(env_name: str, task: TaskBase, enable_gui: bool = True):
    if env_name == "drone":
        return DroneEnv(task, enable_gui)
    elif env_name == "point":
        return PointEnv(task, enable_gui)
    elif env_name == "car":
        return CarEnv(task, enable_gui)
    elif env_name == "doggo":
        return DoggoEnv(task, enable_gui)
    else:
        raise ValueError(f"Env {env_name} not found")
