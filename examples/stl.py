from matplotlib import pyplot as plt

from stl_mob.stl.solver import StlpySolver
from stl_mob.stl.tasks import BranchTask, CoverTask, LoopTask, Map, SequenceTask, SignalTask


def test_seq_task_sol():
    task_map = Map.generate_map(4, (-5, 5), ((1, 1), (2, 2)))
    task = SequenceTask.generate_task(task_map, 4, total_time_steps=10)
    spec = task.get_spec().get_stlpy_form()
    solver = StlpySolver(space_dim=2)
    path, info = solver.solve_stlpy_formula(
        spec, task.sample_init_pos(), task.total_time_steps)
    print(path)
    print(task)
    task.plot_solution(path)
    plt.show()

    print(task.get_quantitative_info())


def test_cover_task_sol():
    task_map = Map.generate_map(4, (-5, 5), ((1, 1), (2, 2)))
    task = CoverTask.generate_task(task_map, 4, total_time_steps=10)
    spec = task.get_spec().get_stlpy_form()
    solver = StlpySolver(space_dim=2)
    path, info = solver.solve_stlpy_formula(
        spec, task.sample_init_pos(), task.total_time_steps)
    print(path)
    print(task)
    task.plot_solution(path)
    plt.show()


def test_branch_task_sol():
    task_map = Map.generate_map(4, (-5, 5), ((1, 1), (2, 2)))
    task = BranchTask.generate_task(task_map, 4, total_time_steps=16)
    spec = task.get_spec().get_stlpy_form()
    solver = StlpySolver(space_dim=2)
    path, info = solver.solve_stlpy_formula(
        spec, task.sample_init_pos(), task.total_time_steps)
    print(path)
    print(task)
    task.plot_solution(path)
    plt.show()


def test_loop_task_sol():
    task_map = Map.generate_map(4, (-5, 5), ((1, 1), (2, 2)))
    task = LoopTask.generate_task(task_map, 4, total_time_steps=16)
    spec = task.get_spec().get_stlpy_form()
    solver = StlpySolver(space_dim=2)
    path, info = solver.solve_stlpy_formula(
        spec, task.sample_init_pos(), task.total_time_steps)
    print(path)
    print(task)
    task.plot_solution(path)
    plt.show()


def test_signal_task_sol():
    task_map = Map.generate_map(4, (-5, 5), ((1, 1), (2, 2)))
    task = SignalTask.generate_task(task_map, 6, total_time_steps=8)
    spec = task.get_spec().get_stlpy_form()
    solver = StlpySolver(space_dim=2)
    path, info = solver.solve_stlpy_formula(
        spec, task.sample_init_pos(), task.total_time_steps)
    print(f"the last second waypoint is: {task.wp_list[-2]}")
    print(f"the last waypoint is: {task.wp_list[-1]}")
    print(path)
    print(task)
    task.plot_solution(path)
    plt.show()


if __name__ == "__main__":
    test_seq_task_sol()
    test_cover_task_sol()
    test_branch_task_sol()
    test_loop_task_sol()
    test_signal_task_sol()
