from stl_mob.envs.wrapper import PointEnv, CarEnv, DoggoEnv, DroneEnv
from stl_mob.stl.tasks import SequenceTask, Map


def test_all_env_render_and_step():
    task_map = Map.generate_map(5,
                                pos_range=((-5, -5), (5, 5)),
                                obs_size_range=((1, 1), (2, 2)),
                                map_size=(10., 10.))
    task = SequenceTask.generate_task(task_map, num_wp=4)
    envs = [
        PointEnv(task),
        CarEnv(task),
        DoggoEnv(task),
        DroneEnv(task),
    ]
    for env in envs:
        env.reset()
        for _ in range(100):
            env.step(env.gym_env.action_space.sample())


if __name__ == '__main__':
    test_all_env_render_and_step()
