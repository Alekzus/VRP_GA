import csv
from datetime import datetime
from operator import attrgetter
from os import makedirs, path
from shutil import rmtree
from typing import Callable, List

from gen.gen import *
from plot.plot import *
from stats.stats import *

NUM_GENERATION: int = 3
NUM_SUB_RUN: int = 20
GENERATION_INTERVAL: int = 100
MIN_GENERATION: int = 100

stats_list: List[dict] = []
generations: List[int] = []


def run_results(
    cross_function: Callable[[List[Individual]], List[Individual]], folder_name: str
):
    base_path: str = f"resultats/{folder_name}"
    with open(path.join(base_path, "result_PM1000_50.csv"), mode="a", newline="") as o:
        writer = csv.writer(o, delimiter=",")
        # writer.writerow(["generations", "run", "start", "end", "best_fitness"])
        for i in range(
            MIN_GENERATION,
            MIN_GENERATION * NUM_GENERATION + GENERATION_INTERVAL,
            GENERATION_INTERVAL,
        ):
            stats: dict = {}
            result_path: str = path.join(base_path, str(i))
            # if path.exists(result_path):
            #     rmtree(result_path)
            # makedirs(result_path)
            print(f"[{folder_name}, {i}] Start {datetime.now().strftime('%H:%M:%S')}")
            for j in range(NUM_SUB_RUN):
                run_start_time: str = datetime.now().strftime("%H:%M:%S")
                print(f"\tRun {j + 1}\tStart\t{run_start_time}")
                result: List[Individual] = run(
                    population_size=50,
                    generation_number=i,
                    cross_function=cross_function,
                )
                run_end_time: str = datetime.now().strftime("%H:%M:%S")
                print(f"\t\tEnd\t{run_end_time}")
                stats[j + 1] = sorted(result[0], key=attrgetter("fitness"))[0].fitness
                # writer.writerow([i, j + 1, run_start_time, run_end_time, stats[j + 1]])
                plot_routes(
                    sorted(result[0], key=attrgetter("fitness"))[0].phenotype,
                    path.join(result_path, f"{j}.png"),
                )
            print(f"[{folder_name}, {i}] End {datetime.now().strftime('%H:%M:%S')}\n")
            stats_list.append(stats)
            generations.append(i)

    # save_stats(
    #     f"resultats/{folder_name}/INV/result_PM1000_50.png", stats_list, generations
    # )


def run_all(n: int = 1):
    algos: List[tuple[Callable[[List[Individual]], List[Individual]], str]] = [
        (order_crossover, "OX"),
        (linear_crossover, "LOX"),
        (position_based_crossover, "POS"),
        (order_based_crossover, "OBX"),
        (heuristic_crossover, "HX"),
        (uniform_crossover, "UX"),
        (edge_recombination_crossover, "ERX"),
        (trajectory_crossover, "TX"),
    ]
    for i in range(n):
        for algo in algos:
            run_results(algo[0], algo[1])
    write_times()
    write_fitness()


# plot_routes(sorted(gen_first_pop(30, 30), key=attrgetter("fitness"))[0].phenotype)
# plot_routes(
#     sorted(
#         run(cross_function=order_crossover, generation_number=1000)[0],
#         key=attrgetter("fitness"),
#     )[0].phenotype
# )
# run_results(uniform_crossover, "UX")
# plot_fitness(path.join("resultats", "fitness.json"))
# plot_times(path.join("resultats", "times.json"))
# plot_scores(path.join("resultats", "scores.json"))
