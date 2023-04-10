from stl_mob.evaluate.plan_and_ctrl_time import TimeEvaluator
from stl_mob.stl.tasks import Map, SequenceTask


def demo():
    task_map = Map.generate_map(5,
                                pos_range=((-3, -3), (3, 3)),
                                obs_size_range=((0.1, 0.1), (0.5, 0.5)),
                                map_size=(10., 10.))
    task = SequenceTask.generate_task(task_map, num_wp=4)

    # The video will show when enable_gui=True
    evaluator = TimeEvaluator(task, "point", enable_gui=True)

    planning_time, ctrl_time = evaluator.eval()

    print(f"STL spec: {task}")
    print(f"obstacles: {task.task_map.obs_list}")
    print(f"waypoints: {task.wp_list}")
    print(f"Planning time: {planning_time: .3f}")
    print(f"Control time steps: {ctrl_time}")


if __name__ == '__main__':
    demo()
