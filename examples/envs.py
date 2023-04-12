from stl_mob.envs.wrapper import PointEnv, CarEnv, DoggoEnv
from stl_mob.envs.wrapper import get_env
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
        # DroneEnv(task),
    ]
    for env in envs:
        env.reset()
        for _ in range(100):
            env.step(env.gym_env.action_space.sample())


def customize_map():
    task_map = Map.generate_map(1,
                                pos_range=((-0.5, -1.5), (-0.5, -1.5)),
                                obs_size_range=((3.0, 0.4), (3.0, 0.4)),
                                map_size=(10., 10.))
    task = SequenceTask.generate_task(task_map, num_wp=0)
    env = get_env("doggo", task)

    env.reset(init_pos=[-3, -3])
    env.add_wp_marker([-2.0, -0.5], 0.3, color=(1, 0, 0, 0.5), alpha=0.5, label="A")
    env.add_wp_marker([3.0, -1.8], 0.3, color=(0, 1, 1, 0.5), alpha=0.5, label="B")
    env.add_wp_marker([2.5, 2.5], 0.3, color=(1, 1, 0, 0.5), alpha=0.5, label="C")
    env.set_goal([-0.5, -1.5])

    while True:
        env.gym_env.render()


if __name__ == '__main__':
    # test_all_env_render_and_step()
    customize_map()
