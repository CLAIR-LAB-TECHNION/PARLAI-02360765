"""The winding-road environment for the Tut03 notebook.

A minimal continuous-control task built to isolate distributional shift: a good
expert keeps the car in a thin tube around the center line, so its
demonstrations cover only a small slice of the state space.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WindingRoadEnv(gym.Env):
    """A car drives at constant speed along a sinusoidal road and must stay on it.

    - Observation: ``(x, y, theta)``, the car's position and heading (radians).
    - Action: a steering rate in ``[-max_steer, max_steer]`` rad/s, as an array of shape ``(1,)``.
    - Wind gusts add Gaussian noise to the steering at every step.
    - The episode terminates when the car leaves the road (failure) or reaches
      ``x = length`` (success, reported as ``info["success"]``). It is truncated
      after ``max_steps`` steps.
    - Reward: forward progress, so the return is the distance driven along x.
    """

    metadata = {"render_modes": []}

    def __init__(self, amplitude=2.0, period=20.0, length=40.0, half_width=1.0, speed=2.0, dt=0.1,
                 max_steer=2.0, gust=0.6, max_steps=400):
        """
        Args:
            amplitude, period: the road's center line is ``y = amplitude * sin(2 pi x / period)``.
            length: the x coordinate of the finish line.
            half_width: the car is off the road when ``|y - center(x)| > half_width``.
            speed: the car's constant speed.
            dt: the duration of one step.
            max_steer: the largest steering rate (rad/s), in either direction.
            gust: the standard deviation of the wind's noise on the steering rate.
            max_steps: the episode is truncated after this many steps.
        """
        super().__init__()
        self.amplitude = amplitude
        self.period = period
        self.length = length
        self.half_width = half_width
        self.speed = speed
        self.dt = dt
        self.max_steer = max_steer
        self.gust = gust
        self.max_steps = max_steps
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
        self.action_space = spaces.Box(-max_steer, max_steer, shape=(1,), dtype=np.float32)
        self.state = None
        self.steps = 0

    def center(self, x):
        """The y coordinate of the road's center line at position x."""
        return self.amplitude * np.sin(2 * np.pi * x / self.period)

    def road_heading(self, x):
        """The direction of the road at position x (radians)."""
        slope = self.amplitude * 2 * np.pi / self.period * np.cos(2 * np.pi * x / self.period)
        return np.arctan(slope)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        y = self.np_random.uniform(-0.2, 0.2)
        theta = self.road_heading(0.0) + self.np_random.uniform(-0.2, 0.2)
        self.state = np.array([0.0, y, theta])
        self.steps = 0
        return self.state.copy(), {"success": False}

    def step(self, action):
        steer = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -self.max_steer, self.max_steer))
        x, y, theta = self.state
        theta = theta + (steer + self.gust * self.np_random.standard_normal()) * self.dt
        new_x = x + self.speed * np.cos(theta) * self.dt
        new_y = y + self.speed * np.sin(theta) * self.dt
        self.state = np.array([new_x, new_y, theta])
        self.steps += 1

        off_road = abs(new_y - self.center(new_x)) > self.half_width
        finished = new_x >= self.length and not off_road
        terminated = off_road or finished
        truncated = self.steps >= self.max_steps and not terminated
        return self.state.copy(), new_x - x, terminated, truncated, {"success": finished}
