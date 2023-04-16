from abc import ABC, abstractmethod
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from stlpy.STL import STLTree

from stl_mob.stl.stl import STL, inside_rectangle_formula, outside_rectangle_formula


class Obstacle:
    def __init__(self, pos: np.ndarray, size: np.ndarray, keepout: float = 0.0):
        self.pos = pos
        self.size = size
        self.keepout = keepout
        self.bound = np.stack([self.pos - (self.size + self.keepout) / 2,
                               self.pos + (self.size + self.keepout) / 2]).T.flatten()

        self.spec = None
        self.neg_spec = None

    def get_spec(self, obs_name: str = None):
        if obs_name is None:
            obs_name = str(self)
        if self.spec is None:
            self.spec = STL(outside_rectangle_formula(self.bound, 0, 1, 2, name=obs_name))
        return self.spec

    def workaround_not(self, obs_name: str = None):
        if obs_name is None:
            obs_name = str(self)
        if self.neg_spec is None:
            self.neg_spec = STL(inside_rectangle_formula(self.bound, 0, 1, 2, name=obs_name))
        return self.neg_spec

    def workaround_implies(self, other: STL, obs_name: str = None):
        neg_spec = self.workaround_not(obs_name=obs_name)
        return neg_spec | other

    def contains(self, pos: np.ndarray):
        return np.all(np.abs(pos - self.pos) < (self.size + self.keepout) / 2 - 1e-3)

    def __repr__(self):
        return f"obs(pos={self.pos}, size={self.size})"


class Waypoint:
    def __init__(self, pos: np.ndarray, size: float):
        self.pos = pos
        self.size = size
        self.bound = np.stack([self.pos - self.size / 2, self.pos + self.size / 2]).T.flatten()
        self.spec = None
        self.neg_spec = None

    def get_spec(self, wp_name: str = None):
        if wp_name is None:
            wp_name = str(self)
        if self.spec is None:
            self.spec = STL(inside_rectangle_formula(self.bound, 0, 1, 2, name=wp_name))
        return self.spec

    def workaround_not(self, wp_name: str = None):
        if wp_name is None:
            wp_name = str(self)
        if self.neg_spec is None:
            self.neg_spec = STL(outside_rectangle_formula(self.bound, 0, 1, 2, name=wp_name))
        return self.neg_spec

    def workaround_implies(self, other: STL, wp_name: str = None):
        neg_spec = self.workaround_not(wp_name=wp_name)
        return neg_spec | other

    def contains(self, pos: np.ndarray):
        return np.all(np.abs(pos - self.pos) < self.size / 2 + 1e-3)

    def __repr__(self):
        return f"wp(pos={self.pos}, size={self.size})"


class Map:
    def __init__(self,
                 obs_list: list[Obstacle],
                 pos_range: tuple,
                 obs_size_range: tuple,
                 map_size: tuple,
                 obs_name_list: str = None,
                 shuffle_obs_name: bool = True):
        self.obs_list = obs_list
        self.pos_range = pos_range
        self.obs_size_range = obs_size_range
        self.map_size = map_size
        self.shuffle_obs_name = shuffle_obs_name

        if obs_name_list is None:
            self.obs_name_list = [f"obs_{i}" for i in range(len(obs_list))]
            if shuffle_obs_name:
                np.random.shuffle(self.obs_name_list)
        else:
            self.obs_name_list = obs_name_list

    @classmethod
    def generate_map(cls, num_obs: int,
                     pos_range: tuple[tuple, tuple],
                     obs_size_range: tuple[tuple, tuple],
                     map_size: tuple[float, float] = (10., 10.)) -> "Map":
        obs_list = []
        for _ in range(num_obs):
            size = np.random.uniform(*obs_size_range)
            pos = np.random.uniform(*pos_range)
            obs = Obstacle(pos, size)
            obs_list.append(obs)
        return cls(obs_list, pos_range, obs_size_range, map_size)

    def get_spec(self) -> STLTree:
        spec = self.obs_list[0].get_spec(obs_name=self.obs_name_list[0])
        for obs, obs_name in zip(self.obs_list[1:], self.obs_name_list[1:]):
            spec = spec & obs.get_spec(obs_name=obs_name)
        return spec

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        ax.set_xlim([-self.map_size[0], self.map_size[0]])
        ax.set_ylim([-self.map_size[1], self.map_size[1]])

        for obs in self.obs_list:
            rect = Rectangle(obs.pos - obs.size / 2, obs.size[0], obs.size[1], color="red", alpha=0.5)
            ax.add_patch(rect)

        return fig, ax

    def inside_obs(self, pos: np.ndarray) -> bool:
        for obs in self.obs_list:
            if obs.contains(pos):
                return True
        return False


class TaskBase(ABC):
    def __init__(self,
                 task_map: Map,
                 wp_list: list[Waypoint],
                 wp_size: float,
                 total_time_steps: int,
                 wp_name_list: str = None,
                 shuffle_wp_name: bool = True):
        self.task_map = task_map
        self.wp_list = wp_list
        self.wp_size = wp_size
        self.total_time_steps = total_time_steps
        self.shuffle_wp_name = shuffle_wp_name
        self.spec = None

        if wp_name_list is None:
            self.wp_name_list = [f"wp_{i}" for i in range(len(wp_list))]
            if shuffle_wp_name:
                np.random.shuffle(self.wp_name_list)
        else:
            self.wp_name_list = wp_name_list

    @classmethod
    def generate_task(cls,
                      task_map: Map,
                      num_wp: int,
                      wp_size: float = 0.5,
                      total_time_steps: int = 20):
        wp_list = []
        for i in range(num_wp):
            pos = np.random.uniform(*task_map.pos_range, size=2)
            while task_map.inside_obs(pos):
                pos = np.random.uniform(*task_map.pos_range, size=2)
            wp = Waypoint(pos, wp_size)
            wp_list.append(wp)
        return cls(task_map, wp_list, wp_size, total_time_steps)

    def plot_map(self):
        fig, ax = self.task_map.plot()
        for wp in self.wp_list:
            rect = Rectangle(wp.pos - wp.size / 2, wp.size, wp.size, color="blue", alpha=0.5)
            ax.add_patch(rect)
        return fig, ax

    def plot_solution(self, path: list):
        fig, ax = self.plot_map()
        ax.scatter([p[0] for p in path], [p[1] for p in path], color="green")

        return fig, ax

    def sample_init_pos(self):
        pos = np.random.uniform(*self.task_map.pos_range, size=2)
        while self.task_map.inside_obs(pos):
            pos = np.random.uniform(*self.task_map.pos_range, size=2)
        return pos

    def get_spec(self) -> STL:
        if self.spec is None:
            self.spec = self._get_spec()
            self.spec.simplify()
        return self.spec

    def get_quantitative_info(self) -> tuple[np.ndarray, np.ndarray]:
        wp_info = np.array([np.r_[wp.pos, wp.size] for wp in self.wp_list])
        obs_info = np.array([np.r_[obs.pos, obs.size] for obs in self.task_map.obs_list])

        return wp_info, obs_info

    def __repr__(self):
        return self.get_spec().__repr__()

    @abstractmethod
    def _get_spec(self) -> STL:
        pass


class SequenceTask(TaskBase):
    def _get_spec(self) -> STL:
        intv = self.total_time_steps // len(self.wp_list)
        seq_spec = self.wp_list[0].get_spec(self.wp_name_list[0]).eventually(0, intv)
        for i in range(1, len(self.wp_list)):
            seq_spec = seq_spec & self.wp_list[i].get_spec(self.wp_name_list[i]).eventually(i * intv, (i + 1) * intv)
        collision_spec = self.task_map.get_spec().always(0, self.total_time_steps)
        spec = seq_spec & collision_spec

        return spec


class CoverTask(TaskBase):
    def _get_spec(self):
        cover_spec = self.wp_list[0].get_spec(self.wp_name_list[0]).eventually(0, self.total_time_steps)
        for i in range(1, len(self.wp_list)):
            cover_spec &= self.wp_list[i].get_spec(self.wp_name_list[i]).eventually(0, self.total_time_steps)
        collision_spec = self.task_map.get_spec().always(0, self.total_time_steps)
        spec = cover_spec & collision_spec

        return spec


class BranchTask(TaskBase):

    @classmethod
    def generate_task(cls,
                      task_map: Map,
                      num_wp: int,
                      wp_size: float = 0.5,
                      total_time_steps: int = 20):
        assert num_wp % 2 == 0
        return super().generate_task(task_map, num_wp, wp_size, total_time_steps)

    def _get_spec(self):
        # each two waypoints are one group, if reach one, must reach the other later
        branch_spec = self.wp_list[0].get_spec(self.wp_name_list[0]).eventually(0, self.total_time_steps // 2) & \
                      self.wp_list[1].get_spec(self.wp_name_list[1]).eventually(self.total_time_steps // 2,
                                                                                self.total_time_steps)

        for i in range(2, len(self.wp_list), 2):
            branch_spec |= self.wp_list[i].get_spec(self.wp_name_list[i]).eventually(0, self.total_time_steps // 2) & \
                           self.wp_list[i + 1].get_spec(self.wp_name_list[i + 1]).eventually(self.total_time_steps // 2,
                                                                                             self.total_time_steps)

        collision_spec = self.task_map.get_spec().always(0, self.total_time_steps)
        spec = branch_spec & collision_spec

        return spec


class LoopTask(TaskBase):
    def _get_spec(self):
        # loop among all waypoints
        loop_spec = self.wp_list[0].get_spec(self.wp_name_list[0]) & self.wp_list[1].get_spec(
            self.wp_name_list[1]).eventually(0, 1)
        for i in range(1, len(self.wp_list)):
            loop_spec |= \
                self.wp_list[i].get_spec(self.wp_name_list[i]) & self.wp_list[(i + 1) % len(self.wp_list)].get_spec(
                    self.wp_name_list[(i + 1) % len(self.wp_list)]).eventually(0, 1)
        loop_spec = loop_spec.always(1, self.total_time_steps - 1)

        collision_spec = self.task_map.get_spec().always(0, self.total_time_steps)
        spec = loop_spec & collision_spec

        return spec


class SignalTask(TaskBase):
    def _get_spec(self):
        # repeatedly visit all waypoints except the last one until visit the last second waypoint,
        # finally reach the last one
        signal_spec = self.wp_list[0].get_spec(self.wp_name_list[0])
        for i in range(1, len(self.wp_list) - 1):
            signal_spec |= self.wp_list[i].get_spec(self.wp_name_list[i])
        signal_spec = signal_spec
        signal_spec = signal_spec.until(self.wp_list[-2].get_spec(self.wp_name_list[-2]), 1, self.total_time_steps - 3)
        signal_spec &= self.wp_list[-1].get_spec(self.wp_name_list[-1]).eventually(0, self.total_time_steps)

        collision_spec = self.task_map.get_spec().always(0, self.total_time_steps)
        spec = signal_spec & collision_spec

        return spec


class AltCoverTask(TaskBase):
    def _get_spec(self):
        # STL spec like F[0,28]B&G[0,28](B=>F[0,28]C)&G[0,28](C=>F[0,28]A)&G[0,28]~obs
        all_wps_specs = [wp.get_spec(self.wp_name_list[i]) for i, wp in enumerate(self.wp_list)]
        all_neg_wps_specs = [wp.workaround_not("~" + self.wp_name_list[i]) for i, wp in enumerate(self.wp_list)]
        better_cover_spec = all_wps_specs[0].eventually(0, self.total_time_steps)
        for i in range(1, len(self.wp_list)):
            better_cover_spec &= all_wps_specs[i].eventually(0, self.total_time_steps)

        seq_spec = reduce(lambda x, y: x & y, all_neg_wps_specs[1:]).until(all_wps_specs[0], 0, self.total_time_steps)
        for i in range(1, len(self.wp_list) - 1):
            seq_spec |= reduce(
                lambda x, y: x & y,
                all_neg_wps_specs[i + 1:]).until(
                all_wps_specs[i], 0, self.total_time_steps)

        better_cover_spec &= seq_spec

        collision_spec = self.task_map.get_spec().always(0, self.total_time_steps)
        spec = better_cover_spec & collision_spec

        return spec
