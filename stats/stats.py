import csv
import json
import os
from datetime import datetime
from statistics import mean
from typing import List

files: List[str] = [
    os.path.join("results", dir, "INV", "result_PM1000_50.csv")
    for dir in os.listdir("results")
    if ".json" not in dir
]


def write_times():
    stats: dict = {}
    for stats_file in files:
        with open(stats_file) as f:
            reader = csv.reader(f)
            next(reader, None)
            algo: str = stats_file.split("\\")[1]
            stats[algo]: dict = {}
            for i in range(100, 1100, 100):
                stats[algo][str(i)]: List[int] = []
            for row in reader:
                start = datetime.strptime(row[2], "%H:%M:%S")
                end = datetime.strptime(row[3], "%H:%M:%S")
                stats[algo][row[0]].append((end - start).seconds)
    for key in stats.keys():
        for gen in stats[key]:
            stats[key][gen] = mean(stats[key][gen])
    with open(os.path.join("results", "times.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)


def write_fitness():
    stats: dict = {}
    for stats_file in files:
        with open(stats_file) as f:
            reader = csv.reader(f)
            next(reader, None)
            algo: str = stats_file.split("\\")[1]
            stats[algo]: dict = {}
            for i in range(100, 1100, 100):
                stats[algo][str(i)]: List[int] = []
            for row in reader:
                stats[algo][row[0]].append(float(row[4]))
    for key in stats.keys():
        for gen in stats[key]:
            stats[key][gen] = mean(stats[key][gen])
    with open(os.path.join("results", "fitness.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)


def write_scores():
    scores: dict = {}
    with open(os.path.join("results", "times.json"), "r") as t, open(
        os.path.join("results", "fitness.json"), "r"
    ) as f:
        times: dict = json.load(t)
        fitness: dict = json.load(f)
        for algo in times.keys():
            scores[algo]: dict = {}
            for gen in times[algo]:
                scores[algo][gen] = fitness[algo][gen] * times[algo][gen]
    with open(os.path.join("results", "scores.json"), "w", encoding="utf-8") as s:
        json.dump(scores, s, ensure_ascii=False, indent=4)
