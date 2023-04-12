import os

import numpy as np
from gym.wrappers import Monitor

from stl_mob.ctrl.wrapper import get_controller
from stl_mob.envs.wrapper import get_env
from stl_mob.stl.solver import StlpySolver
from stl_mob.stl.tasks import TaskBase
from stl_mob.utils import DATA_DIR


class TimeEvaluator:
    def __init__(self, task: TaskBase, robot_name: str, enable_gui: bool = True, store_video: bool = False):
        self.space_dims = {
            "point": 2,
            "car": 2,
            "doggo": 2,
            "drone": 3
        }

        self.task = task
        self.robot_name = robot_name
        self.enable_gui = enable_gui
        self.store_video = store_video

        self.env = get_env(self.robot_name, self.task, self.enable_gui)
        self.solver = StlpySolver(space_dim=self.space_dims[self.robot_name])
        self.controller = get_controller(self.robot_name)

    def plan(self):
        path, info = self.solver.solve_stlpy_formula(self.task.get_spec().get_stlpy_form(),
                                                     x0=self.task.sample_init_pos(),
                                                     total_time=self.task.total_time_steps)

        solve_time = info["solve_t"]
        return path, solve_time

    def ctrl(self, path: np.ndarray, max_wp_step: int = 300) -> float:
        total_time = 0

        if self.store_video:
            save_dir = f"{DATA_DIR}/videos/{self.robot_name}_{type(self.task).__name__.lower()}"
            os.makedirs(save_dir, exist_ok=True)
            self.env.gym_env = Monitor(self.env.gym_env, save_dir, force=True)

        obs = self.env.reset(init_pos=path[0])
        for wp in path[1:]:
            self.env.set_goal(wp)
            t = 0
            for t in range(max_wp_step):
                action = self.controller(obs)
                obs, _, _, _ = self.env.step(action)
                if self.env.reached():
                    break
            total_time += t
            if not self.env.reached():
                return float('inf')

        return total_time

    def eval(self):
        path, solve_t = self.plan()

        if path is None:
            return float('inf'), float('inf')

        ctrl_time = self.ctrl(path)
        return solve_t, ctrl_time
