"""Plotting helpers for the Tut02 notebook (Markov Models).

Everything here is presentation-only: distribution trajectories, long-run time
fractions, belief heatmaps, and the parameter-count figure. The models and
algorithms the tutorial teaches (propagation, sampling, belief filtering,
inference by enumeration, flattening a DBN) live in the notebook cells.

Colors follow the entity: each room keeps the same color in every figure.
The three room colors are a colorblind-safe categorical set (validated on all
pairs), and every line also carries its own marker so identity never depends
on color alone.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# one fixed color and marker per room: mailroom, office1, office2
ROOM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
ROOM_MARKERS = ["o", "s", "^"]

# neutral inks for comparisons that are not about a specific room
INK = "#0b0b0b"
MUTED = "#8a8986"


def plot_distribution_evolution(mus, state_names, stationary=None, ax=None, title=None):
    """Plot the state distribution mu_t over time, one line per state.

    Args:
        mus: array of shape (T + 1, n) whose row t is the distribution mu_t.
        state_names: list of n state names (used for the legend).
        stationary: optional length-n stationary distribution, drawn as dashed
            horizontal reference lines in the matching state colors.
        ax: axes to draw on (defaults to current axes).
        title: optional axes title.
    """
    ax = ax or plt.gca()
    mus = np.asarray(mus)
    t = np.arange(len(mus))
    for i, name in enumerate(state_names):
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        ax.plot(t, mus[:, i], color=color, marker=ROOM_MARKERS[i % 3], markersize=6,
                linewidth=2, label=name)
        if stationary is not None:
            ax.axhline(stationary[i], color=color, linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"$\mu_t[i] = P(X_t = i)$")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(title="state $i$" + (" (dashed: stationary)" if stationary is not None else ""))
    if title:
        ax.set_title(title)


def plot_time_fractions(states, stationary, state_names, ax=None):
    """Plot the running fraction of time a sampled trajectory spent in each state.

    Args:
        states: 1-D integer array, the sampled trajectory x_0, x_1, ..., x_T.
        stationary: length-n stationary distribution, drawn as dashed lines.
        state_names: list of n state names (used for the legend).
        ax: axes to draw on (defaults to current axes).
    """
    ax = ax or plt.gca()
    states = np.asarray(states)
    t = np.arange(1, len(states) + 1)
    for i, name in enumerate(state_names):
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        running = np.cumsum(states == i) / t
        ax.plot(t, running, color=color, linewidth=2, label=name,
                marker=ROOM_MARKERS[i % 3], markevery=0.1, markersize=6)
        ax.axhline(stationary[i], color=color, linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("number of steps simulated")
    ax.set_ylabel("fraction of time in state")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(title="dashed: stationary")


def plot_start_comparison(curves_by_start, state_name, ax=None, title=None):
    """Plot P(X_t = state) over time for several starting states.

    Args:
        curves_by_start: dict mapping a starting-state name to a 1-D array of
            P(X_t = state) for t = 0, 1, ..., T. Starting states must be room
            names, so each curve takes that room's color.
        state_name: name of the tracked state (used in the y-label).
        ax: axes to draw on (defaults to current axes).
        title: optional axes title.
    """
    ax = ax or plt.gca()
    for i, (start, curve) in enumerate(curves_by_start.items()):
        curve = np.asarray(curve)
        ax.plot(np.arange(len(curve)), curve, color=ROOM_COLORS[i % 3],
                marker=ROOM_MARKERS[i % 3], markersize=5, linewidth=2,
                label=f"start in {start}")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel(f"$P(X_t = $ {state_name}$)$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)


def plot_distribution_bars(dists, state_names, ax=None, label_state=0):
    """Grouped bar chart comparing several distributions over the same states.

    Args:
        dists: dict mapping a group label (e.g. a rule name) to a length-n
            distribution.
        state_names: list of n state names (used for the legend).
        ax: axes to draw on (defaults to current axes).
        label_state: index of the state whose bars get value labels (the
            quantity under discussion); None disables labels.
    """
    ax = ax or plt.gca()
    groups = list(dists)
    n = len(state_names)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, name in enumerate(state_names):
        heights = [dists[g][i] for g in groups]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, heights, width=width * 0.92,
                      color=ROOM_COLORS[i % 3], label=name)
        if label_state == i:
            ax.bar_label(bars, labels=[f"{h:.1%}" for h in heights], padding=2, fontsize=9)
    ax.set_xticks(x, groups)
    ax.set_ylabel("long-run fraction of time")
    ax.set_ylim(0, max(max(d) for d in dists.values()) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()


def plot_belief(beliefs, true_states, state_names, ax=None, title=None, colorbar=True):
    """Heatmap of a belief over time, with the true (hidden) state overlaid.

    Args:
        beliefs: array of shape (T + 1, n); row t is the belief over the n
            values at time t.
        true_states: 1-D integer array of length T + 1 with the true values.
        state_names: list of n value names (used as y tick labels).
        ax: axes to draw on (defaults to current axes).
        title: optional axes title.
        colorbar: whether to attach a colorbar.
    """
    ax = ax or plt.gca()
    beliefs = np.asarray(beliefs)
    im = ax.imshow(beliefs.T, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest", origin="lower")
    t = np.arange(len(true_states))
    ax.scatter(t, true_states, s=22, facecolor="white", edgecolor=INK, linewidth=1,
               label="true (hidden) state", zorder=3)
    ax.set_yticks(range(len(state_names)), state_names)
    ax.set_xlabel("time step $t$")
    # legend above the top-right corner, so it never hides belief cells
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False,
              borderaxespad=0.2)
    if colorbar:
        plt.colorbar(im, ax=ax, label="belief", pad=0.01)
    if title:
        ax.set_title(title)


def plot_parameter_scaling(n_vars, flat_sizes, factored_sizes, ax=None):
    """Log-scale comparison of flat vs factored transition-model sizes.

    Args:
        n_vars: 1-D array with the number of state variables.
        flat_sizes: number of entries in the flat transition matrix for each
            entry of ``n_vars``.
        factored_sizes: number of entries in all conditional probability
            tables of the factored (DBN) model for each entry of ``n_vars``.
        ax: axes to draw on (defaults to current axes).
    """
    ax = ax or plt.gca()
    ax.plot(n_vars, flat_sizes, color=INK, marker="o", markersize=6, linewidth=2,
            label="flat transition matrix")
    ax.plot(n_vars, factored_sizes, color=ROOM_COLORS[0], marker="s", markersize=6,
            linewidth=2, label="factored (DBN) tables")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("number of binary state variables $m$")
    ax.set_ylabel("numbers to specify the transitions")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
