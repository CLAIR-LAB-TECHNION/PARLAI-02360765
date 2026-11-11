"""Utility code for Tutorial 03 (Imitation Learning).

This package holds the non-pedagogical scaffolding for the notebook: the Taxi
expert and start-state wrapper, the winding-road environment, and the plotting
helpers. What the tutorial actually teaches (collecting demonstrations,
behavior cloning, evaluation, DAgger, data augmentation) lives in the notebook
cells themselves.
"""

from .plotting import (
    AUG_COLOR,
    BC_COLOR,
    DAGGER_COLOR,
    EXPERT_COLOR,
    MUTED,
    plot_augmentation,
    plot_cars,
    plot_data_curve,
    plot_road,
    plot_success_vs_labels,
    plot_trajectories,
)
from .road import WindingRoadEnv
from .taxi import TAXI_ACTIONS, TaxiExpert, make_taxi, taxi_frames
