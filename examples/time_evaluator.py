from stl_mob.evaluate.plan_and_ctrl_time import TimeEvaluator
from stl_mob.stl.tasks import Map, SequenceTask, CoverTask, BranchTask, LoopTask, SignalTask


def demo(task_name="seq", robot_name="doggo"):
    task_map = Map.generate_map(4,
                                pos_range=((-3, -3), (3, 3)),
                                obs_size_range=((0.1, 0.1), (0.5, 0.5)),
                                map_size=(10., 10.))
    if task_name == "seq":
        task = SequenceTask.generate_task(task_map, num_wp=4)
    elif task_name == "cover":
        task = CoverTask.generate_task(task_map, num_wp=4)
    elif task_name == "branch":
        task = BranchTask.generate_task(task_map, num_wp=4)
    elif task_name == "loop":
        task = LoopTask.generate_task(task_map, num_wp=4)
    elif task_name == "signal":
        task = SignalTask.generate_task(task_map, num_wp=4)
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    # The video will show when enable_gui=True
    evaluator = TimeEvaluator(task, robot_name, enable_gui=False, store_video=True)

    planning_time, ctrl_time = evaluator.eval()

    print(f"STL spec: {task}")
    print(f"obstacles: {task.task_map.obs_list}")
    print(f"waypoints: {task.wp_list}")
    print(f"Planning time: {planning_time: .3f}")
    print(f"Control time steps: {ctrl_time}")


if __name__ == '__main__':
    demo()
