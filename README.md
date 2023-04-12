# STL Mobile Robot

Turn STL formulas into maps and planed paths, control robots with DRL controllers.

## Use Case

Giving following specification:

```
STL spec: (((F[0, 5]wp_0&F[5, 10]wp_2)&F[10, 15]wp_1)&F[15, 20]wp_3)&G[0, 20](((obs_3&obs_0)&obs_1)&obs_2)
obstacles: [obs(pos=[ 2.30221051 -0.03626683], size=[0.39760751 0.12037159]), obs(pos=[-1.69138811  0.3498487 ],
size=[0.38951315 0.34430025]), obs(pos=[1.07275921 1.93284017], size=[0.42257434 0.37814822]), obs(
pos=[0.70020247 0.53295503], size=[0.28787348 0.25667014])]
waypoints: [wp(pos=[-2.33267981  2.48727927], size=0.5), wp(pos=[-0.15734173 -1.19723772], size=0.5), wp(
pos=[-2.73984922 -0.14574555], size=0.5), wp(pos=[2.0591308  1.65148679], size=0.5)]
```

We can get the following map and control trajectory:


https://user-images.githubusercontent.com/73256697/231562114-f19982c7-94fb-4b26-856a-1fbda30975b7.mp4


With planning time 2.755s and control time steps 760.
```
Planning time:  2.755
Control time steps: 760
```

## Installation

1. Install this package and some dependencies.

```commandline
pip install -e .
```

2. Configure MuJoCo-py. See [here](https://github.com/openai/mujoco-py) for more details.
3. Get a gurobi license and place it
   in [right place](https://www.gurobi.com/documentation/9.5/remoteservices/licensing.html#:~:text=When%20you%20download%20the%20license,%2FLibrary%2Fgurobi%2F%20on%20macOS).
   It is free for academic use. See [here](https://www.gurobi.com/) for more details.

## Task Templates

Check out [tasks.py](./examples/time_evaluator.py) and [stl.py](./examples/stl.py) for more details.

## Planning

Planning is backend by [stlpy](https://stlpy.readthedocs.io/en/latest/). Check out [stl.py](./examples/stl.py) for more
details.

## Control

Controller is a goal-conditioned DRL agent. Check out [control.py](./examples/control.py) for more details.

## Robot Dynamics

Suppose 4 robot dynamics models are available. Check out [envs.py](./examples/envs.py) for more details.
