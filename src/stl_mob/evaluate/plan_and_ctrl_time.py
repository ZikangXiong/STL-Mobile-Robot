import numpy as np

from stl_mob.ctrl.wrapper import get_controller
from stl_mob.envs.wrapper import get_env
from stl_mob.stl.solver import StlpySolver
from stl_mob.stl.tasks import TaskBase


class TimeEvaluator:
    def __init__(self, task: TaskBase, robot_name: str, enable_gui: bool = True):
        self.space_dims = {
            "point": 2,
            "car": 2,
            "doggo": 2,
            "drone": 3
        }

        self.task = task
        self.robot_name = robot_name
        self.enable_gui = enable_gui

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
        obs = self.env.reset(init_pos=path[0])
        total_time = 0
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
