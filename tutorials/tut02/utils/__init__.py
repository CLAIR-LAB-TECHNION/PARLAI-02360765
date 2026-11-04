"""Utility code for Tutorial 02 (Markov Models).

This package holds the non-pedagogical scaffolding for the notebook: the
plotting helpers. The models and algorithms this tutorial actually teaches
(distribution propagation, sampling, belief filtering, inference by
enumeration, flattening a dynamic Bayesian network) live in the notebook
cells themselves.
"""

from .plotting import (
    ROOM_COLORS,
    plot_belief,
    plot_distribution_bars,
    plot_distribution_evolution,
    plot_parameter_scaling,
    plot_start_comparison,
    plot_time_fractions,
)
