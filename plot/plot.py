import itertools
import json
from typing import List

import matplotlib.cm as cm
import matplotlib.patches as ptch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path

from data.data import Client
from sweep.sweep import distance_route


def plot_routes(routes: List[List[Client]], image_name: str = None):
    fig, ax = plt.subplots()

    colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, len(routes))))
    handles = []

    for route in routes:
        verts = []
        codes = []
        color = next(colors)
        for client in route:
            verts.append((client.x, client.y))
            if client.name == "a":
                codes.append(Path.MOVETO)
            else:
                codes.append(Path.LINETO)
                ax.annotate(client.name, (client.x, client.y))
            ax.scatter(client.x, client.y, zorder=2, color=color)
        verts.append((route[0].x, route[0].y))
        codes.append(Path.CLOSEPOLY)
        path = Path(verts, codes)
        patch = ptch.PathPatch(
            path, edgecolor=color, fill=False, label=str(distance_route(route))
        )
        handles.append(patch)
        ax.add_patch(patch)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    plt.legend(handles=handles)
    if image_name:
        plt.savefig(image_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close("all")


def save_stats(image_name: str, stats: List[dict], generations: List[int]):
    fig = plt.figure()
    lines = []
    labels = []
    x = stats[0].keys()
    colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, len(stats))))
    for idx, stat in enumerate(stats):
        y = stat.values()
        color = next(colors)
        plt.plot(x, y, color=color, label=generations[idx])
    for ax in fig.axes:
        ax_lines, ax_labels = ax.get_legend_handles_labels()
        lines.extend(ax_lines)
        labels.extend(ax_labels)
    fig.legend(lines, labels, loc="upper right")
    plt.savefig(image_name, bbox_inches="tight")
    plt.close()


def plot_times(times_file: str):
    with open(times_file) as f:
        times = json.load(f)
        df = pd.DataFrame(times)
        df.plot(xlabel="Number of generations", ylabel="Seconds", grid=True)
        plt.show()
        plt.close("all")


def plot_fitness(fitness_file: str):
    with open(fitness_file) as f:
        times = json.load(f)
        df = pd.DataFrame(times)
        df.plot(xlabel="Number of generations", ylabel="Fitness", grid=True)
        df.plot(kind="box", ylabel="Fitness")
        plt.show()
        plt.close("all")


def plot_scores(scores_file: str):
    with open(scores_file) as f:
        scores = json.load(f)
        df = pd.DataFrame(scores)
        df.plot(xlabel="Number of generations", ylabel="Score", grid=True)
        plt.show()
        plt.close("all")
