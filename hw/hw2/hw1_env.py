"""Your HW1 ``CalderaEnv``, reused in this assignment.

Before you start, replace the ``CalderaEnv`` class below with the ``CalderaEnv``
class from your ``hw1/to_implement.py`` (Task 1 of HW1). If your class uses any
helper functions or imports that you added in HW1, copy them as well. The
reward functions, ``SCalderaEnv``, and ``POCalderaEnv`` from HW1 are not used
in this assignment.

This file is NOT submitted. When we grade this assignment, we use our reference
implementation of HW1 instead of this file, so mistakes in your HW1 do not
affect your grade here. However, if your HW1 environment has bugs, your agents
will be developed and tested on a buggy environment, so fix them first.
"""

from typing import Callable, Optional, Tuple, Union  # noqa: F401

from gymnasium import spaces  # noqa: F401
import numpy as np  # noqa: F401

from caldera_env import (  # noqa: F401
    BaseCalderaEnv,
    DIRECTION_STEPS,
    MOVEMENT_ACTIONS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_WEST,
    SAMPLE,
    validate_bounds,
)
from utils import generate_path  # noqa: F401


class CalderaEnv(BaseCalderaEnv):
    """Replace this class with the ``CalderaEnv`` class from your HW1 ``to_implement.py``."""
