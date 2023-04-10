import numpy as np

from stl_mob.ctrl.wrapper import get_controller
from stl_mob.envs.wrapper import get_env
from stl_mob.stl.solver import StlpySolver
from stl_mob.stl.tasks import SequenceTask, Map


def test_pnc():
    task_map = Map.generate_map(5,
                                pos_range=((-3, -3), (3, 3)),
                                obs_size_range=((0.1, 0.1), (0.5, 0.5)),
                                map_size=(10., 10.))
    task = SequenceTask.generate_task(task_map, num_wp=4)
    robots = [
        'point',
        'car',
        'doggo',
        # 'drone'
    ]

    solver = StlpySolver(space_dim=2)
    spec = task.get_spec().get_stlpy_form()
    path, info = solver.solve_stlpy_formula(spec, task.sample_init_pos(), task.total_time_steps)

    for robot in robots:
        env = get_env(robot, task)
        controller = get_controller(robot)

        obs = env.reset()
        env.set_goal(np.array([2, 2]))
        for wp in path:
            env.set_goal(wp)
            for _ in range(1000):
                action = controller(obs)
                obs, _, _, _ = env.step(action)
                if env.reached():
                    break
            if not env.reached():
                print("Failed to reach goal in 1000 steps")
                break


if __name__ == '__main__':
    test_pnc()
