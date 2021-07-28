import csv
import random as rd
from typing import List

import numpy as np


class Client:
    name: str
    x: float
    y: float
    angle: float
    demand: int

    def __init__(self, name: str, x: float, y: float, demand: int, offset: float = 0.0):
        self.name = name
        self.x = x
        self.y = y
        angle = np.arctan2(y, x) - offset
        self.angle = angle if angle > 0 else angle + 2 * np.pi
        self.demand = demand

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, o) -> bool:
        return self.name == o.name

    def __hash__(self) -> int:
        return hash(repr(self))


def write_clients(n, outfile="data\clients.csv") -> None:
    with open(outfile, "w", newline="") as o:
        writer = csv.writer(o, delimiter=",")
        writer.writerows([["client", "x", "y", "demand"], ["a", "0", "0", "0"]])
        for i in range(1, n):
            writer.writerow(
                [i, rd.randint(-5, 5), rd.randint(-5, 5), rd.randrange(5, 20, 5)]
            )


def read_clients(offset: float = 0.0) -> List[Client]:
    clients: List[Client] = []
    with open("data\clients.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            clients.append(
                Client(row[0], float(row[1]), float(row[2]), int(row[3]), offset)
            )
    return clients
