"""Plotting helpers for the Tut03 notebook (Imitation Learning).

Everything here is presentation-only: the road and its trajectories, and the
success-rate curves. The algorithms the tutorial teaches live in the notebook.

Colors follow the entity: the expert is always ink, behavior cloning orange,
DAgger blue, and data augmentation green, in every figure. The three hues are
the colorblind-safe categorical set from Tut02 (validated on all pairs), and
every series also carries its own marker, so identity never depends on color alone.
"""

import matplotlib.pyplot as plt
import numpy as np

EXPERT_COLOR = "#0b0b0b"   # ink
BC_COLOR = "#eb6834"
DAGGER_COLOR = "#2a78d6"
AUG_COLOR = "#1baf7a"
MUTED = "#8a8986"
ROAD_COLOR = "#e6e5e1"

METHOD_STYLES = {
    "expert": dict(color=EXPERT_COLOR, marker="D"),
    "behavior cloning": dict(color=BC_COLOR, marker="o"),
    "DAgger": dict(color=DAGGER_COLOR, marker="s"),
    "data augmentation": dict(color=AUG_COLOR, marker="^"),
}


def plot_road(env, ax=None):
    """Draw the winding road of a `WindingRoadEnv`: the drivable band and its center line."""
    ax = ax or plt.gca()
    x = np.linspace(0, env.length, 400)
    center = env.center(x)
    ax.fill_between(x, center - env.half_width, center + env.half_width, color=ROAD_COLOR, linewidth=0)
    ax.plot(x, center, color="white", linewidth=1.5, linestyle=(0, (6, 6)))  # lane marking
    ax.axvline(env.length, color=MUTED, linewidth=1)
    ax.text(env.length - 0.4, env.amplitude + env.half_width + 0.3, "finish", color=MUTED,
            ha="right", va="bottom", fontsize=9, clip_on=True)   # hidden when zoomed in
    ax.set_xlim(-0.5, env.length + 1)
    ax.set_ylim(-env.amplitude - env.half_width - 0.8, env.amplitude + env.half_width + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_trajectories(runs, color, label, ax=None, linewidth=1.5, alpha=0.6):
    """Draw the (x, y) path of each episode; mark where failed episodes left the road.

    Args:
        runs: list of episodes as returned by the notebook's `run_episode`,
            i.e., tuples ``(observations, actions, total_reward, info)``.
        color: line color (one per method, see the module docstring).
        label: legend label (shown once for the whole group).
    """
    ax = ax or plt.gca()
    n_failed = 0
    for i, (observations, _, _, info) in enumerate(runs):
        path = np.array(observations)
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth, alpha=alpha,
                solid_capstyle="round", label=label if i == 0 else None)
        if not info["success"]:
            n_failed += 1
            ax.plot(path[-1, 0], path[-1, 1], marker="x", color=color, markersize=8, markeredgewidth=2)
    if n_failed:
        ax.plot([], [], linestyle="none", marker="x", color=MUTED, markersize=8, markeredgewidth=2,
                label="left the road")
    _road_legend(ax)


def plot_cars(poses, color, label=None, ax=None, length=0.5):
    """Draw each car pose ``(x, y, theta)`` as an arrow from its position along its heading."""
    ax = ax or plt.gca()
    poses = np.atleast_2d(np.asarray(poses, dtype=float))
    x, y, theta = poses[:, 0], poses[:, 1], poses[:, 2]
    ax.quiver(x, y, np.cos(theta), np.sin(theta), angles="xy", scale_units="xy", scale=1 / length,
              color=color, width=0.004, headwidth=4, headlength=4.5, headaxislength=4, zorder=3)
    ax.plot(x, y, linestyle="none", marker="o", markersize=3, color=color, zorder=3)
    if label:
        ax.plot([], [], linestyle="none", marker=r"$\rightarrow$", markersize=12, color=color, label=label)
        _road_legend(ax)


def plot_augmentation(env, demo_states, demo_actions, synthetic_states, synthetic_actions, ax=None,
                      duration=0.5):
    """Show how data augmentation creates labeled states from demonstrated ones.

    Draws each demonstrated state and the synthetic states made from it (linked by a
    dashed line). From every state, a dotted curve shows its steering label: where the
    car would go if it held that steering rate for `duration` seconds (without wind),
    with one dot per time step. Zooms in on the drawn states.

    Args:
        env: the `WindingRoadEnv`, for its speed and time step.
        demo_states: array of shape ``(n, 3)``, demonstrated poses ``(x, y, theta)``.
        demo_actions: array of shape ``(n,)`` or ``(n, 1)``, the demonstrated steering.
        synthetic_states: array of shape ``(k, n, 3)``, k synthetic poses per demonstrated pose.
        synthetic_actions: array of shape ``(k, n)`` or ``(k, n, 1)``, their steering labels.
    """
    ax = ax or plt.gca()
    demo_states = np.asarray(demo_states, dtype=float)
    synthetic_states = np.asarray(synthetic_states, dtype=float)
    for copies in synthetic_states:
        for demo, synthetic in zip(demo_states, copies):
            ax.plot([demo[0], synthetic[0]], [demo[1], synthetic[1]], color=MUTED, linewidth=0.8,
                    linestyle=(0, (2, 2)), zorder=2)
    n_steps = round(duration / env.dt)
    drawn = []
    groups = [(demo_states, demo_actions, EXPERT_COLOR, "demonstrated steering"),
              (synthetic_states.reshape(-1, 3), synthetic_actions, AUG_COLOR, "steering label")]
    for poses, actions, color, name in groups:
        actions = np.asarray(actions, dtype=float).reshape(-1)
        for i, ((x, y, theta), steer) in enumerate(zip(poses, actions)):
            path = [(x, y)]
            for _ in range(n_steps):   # hold the steering rate, one time step at a time
                theta += steer * env.dt
                x, y = x + env.speed * np.cos(theta) * env.dt, y + env.speed * np.sin(theta) * env.dt
                path.append((x, y))
            path = np.array(path)
            ax.plot(path[1:, 0], path[1:, 1], linestyle="none", marker="o", markersize=2.5, color=color,
                    label=f"{name}, held for {duration:g} s (one dot per step)" if i == 0 else None)
            drawn.append(path)
    plot_cars(demo_states, color=EXPERT_COLOR, label="demonstrated state", ax=ax)
    plot_cars(synthetic_states.reshape(-1, 3), color=AUG_COLOR, label="synthetic state", ax=ax)
    drawn = np.concatenate(drawn)
    ax.set_xlim(drawn[:, 0].min() - 0.4, drawn[:, 0].max() + 0.4)
    ax.set_ylim(drawn[:, 1].min() - 0.4, drawn[:, 1].max() + 0.4)
    _road_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1)


def _road_legend(ax, **kwargs):
    """A legend with one entry per label (e.g., a single "left the road")."""
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    options = {**dict(loc="lower left", fontsize=9, frameon=False, ncol=3), **kwargs}
    ax.legend(unique.values(), unique.keys(), **options)


def plot_success_vs_labels(results, ax=None, title=None):
    """Success rate against the number of expert labels, one line per method.

    Args:
        results: dict mapping a method name (a key of `METHOD_STYLES`) to a list
            with one curve per seed; a curve is a list of (n_labels, success_rate)
            points. Thin lines show the seeds, the thick line their mean.
    """
    ax = ax or plt.gca()
    for name, curves in results.items():
        style = METHOD_STYLES[name]
        curves = np.array(curves, dtype=float)  # (n_seeds, n_points, 2)
        for curve in curves:
            ax.plot(curve[:, 0], curve[:, 1], color=style["color"], linewidth=1, alpha=0.25)
        mean = curves.mean(axis=0)
        ax.plot(mean[:, 0], mean[:, 1], color=style["color"], marker=style["marker"], markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, linewidth=2,
                label=f"{name} (mean of {len(curves)} seeds)")
    ax.set_xlabel("expert labels (state-action pairs)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    if title:
        ax.set_title(title)


def plot_data_curve(n_demos, metrics, ax=None, title=None):
    """Plot metrics of behavior cloning against the number of expert demonstrations.

    Args:
        n_demos: list of dataset sizes (number of demonstrated episodes).
        metrics: dict mapping a metric name to a list of values, one per dataset size.
    """
    ax = ax or plt.gca()
    styles = [dict(color=MUTED, marker="s"), dict(color=BC_COLOR, marker="o")]
    for (name, values), style in zip(metrics.items(), styles):
        ax.plot(n_demos, values, linewidth=2, markersize=7, markeredgecolor="white",
                markeredgewidth=1.5, label=name, **style)
    ax.set_xscale("log")
    ax.set_xticks(n_demos)
    ax.set_xticklabels([str(n) for n in n_demos])
    ax.minorticks_off()
    ax.set_xlabel("number of expert demonstrations (episodes)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    if title:
        ax.set_title(title)
