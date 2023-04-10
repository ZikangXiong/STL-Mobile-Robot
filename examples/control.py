import time

import numpy as np

from stl_mob.ctrl.wrapper import get_controller
from stl_mob.envs.wrapper import get_env
from stl_mob.stl.tasks import SequenceTask, Map


def test_controller():
    task_map = Map.generate_map(5,
                                pos_range=((-5, -5), (5, 5)),
                                obs_size_range=((1, 1), (2, 2)),
                                map_size=(10., 10.))
    task = SequenceTask.generate_task(task_map, num_wp=4)
    robots = [
        'point',
        'car',
        'doggo',
        # 'drone'
    ]

    for robot in robots:
        env = get_env(robot, task)
        controller = get_controller(robot)

        obs = env.reset()
        env.set_goal(np.array([2, 2]))
        for _ in range(200):
            time.sleep(0.01)
            action = controller(obs)
            obs, _, _, _ = env.step(action)


if __name__ == '__main__':
    test_controller()
