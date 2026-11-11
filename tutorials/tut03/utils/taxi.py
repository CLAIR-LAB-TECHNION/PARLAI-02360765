"""Taxi helpers for the Tut03 notebook: the expert, a start-state option, and video frames.

The expert is a planner. Planning is taught in later tutorials, so here it is a
black box that the notebook only queries: give it a state, receive an action.
"""

import gymnasium as gym
import numpy as np

TAXI_ACTIONS = ["south", "north", "east", "west", "pickup", "dropoff"]


class TaxiWrapper(gym.Wrapper):
    """Taxi with two conveniences.

    - ``reset(options={"start_state": s})`` starts the episode from the chosen state
      (taxi position, passenger location, and destination), so a policy can be
      tested from every possible start state.
    - ``info["success"]`` is ``True`` on the step that delivers the passenger.
    """

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if options is not None and "start_state" in options:
            taxi = self.env.unwrapped
            obs = int(options["start_state"])
            taxi.s = obs
            taxi.lastaction = None
            info = {"prob": 1.0, "action_mask": taxi.action_mask(obs)}
        info["success"] = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["success"] = terminated  # Taxi terminates only on a successful dropoff
        return obs, reward, terminated, truncated, info


def make_taxi(**kwargs):
    """Create the (deterministic) Taxi-v4 environment with the conveniences of `TaxiWrapper`."""
    return TaxiWrapper(gym.make("Taxi-v4", **kwargs))


class TaxiExpert:
    """An optimal taxi driver.

    Taxi is deterministic and has only 500 states, so we can compute the number of
    steps from every state to a successful delivery by value iteration (dynamic
    programming) over the known transition table, and always take an action that
    starts a shortest route. With a reward of -1 per step, the value of a state is
    minus its number of steps to go. Ties are broken by the fixed action order, so
    the expert is deterministic.
    """

    def __init__(self, env):
        taxi = env.unwrapped
        n_states, n_actions = taxi.observation_space.n, taxi.action_space.n
        next_state = np.zeros((n_states, n_actions), dtype=int)
        delivers = np.zeros((n_states, n_actions), dtype=bool)
        for s in range(n_states):
            for a in range(n_actions):
                assert len(taxi.P[s][a]) == 1, "the expert assumes deterministic Taxi (is_rainy=False)"
                (_, s_next, _, terminated), = taxi.P[s][a]
                next_state[s, a], delivers[s, a] = s_next, terminated

        # steps_to_go[s] = length of the shortest route from s to a delivery
        steps_to_go = np.full(n_states, np.inf)
        while True:
            # one step, plus the rest of the route (nothing left after a delivery)
            route_length = np.where(delivers, 1.0, 1.0 + steps_to_go[next_state])
            updated = route_length.min(axis=1)
            if np.array_equal(updated, steps_to_go):
                break
            steps_to_go = updated

        self.steps_to_go = steps_to_go
        self.table = route_length.argmin(axis=1)

    def __call__(self, state):
        return int(self.table[state])


def taxi_frames(env, policy, seed=None, start_state=None, max_steps=30):
    """Render one episode of `policy` (at most `max_steps` steps) as a list of RGB frames."""
    options = None if start_state is None else {"start_state": start_state}
    obs, _ = env.reset(seed=seed, options=options)
    frames = [env.render()]
    for _ in range(max_steps):
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        frames.append(env.render())
        if terminated or truncated:
            break
    return frames
