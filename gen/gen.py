import copy as cp
import random as rd
from math import sqrt
from operator import attrgetter
from typing import Callable, List

import numpy as np

from data.data import Client, read_clients
from sweep.sweep import distance_clients, distance_route, sort_clients

DEFAULT_TRUCK_CAPACITY: int = 30

DEFAULT_POP_SIZE: int = 30
DEFAULT_GENERATION_NUMBER: int = 20
DEFAULT_MUTATION_RATE: float = 0.001

INIT_POP_FIRST_THRESHOLD: float = 0.4
INIT_POP_SECOND_THRESHOLD: float = 0.8


def average(phenotype: List[List[Client]]) -> float:
    sum_distance: int = 0
    for route in phenotype:
        sum_distance += distance_route(route)
    return sum_distance / len(phenotype)


def std_dev(phenotype: List[List[Client]]) -> float:
    avg: float = average(phenotype)
    sum_dev: float = 0.0
    for route in phenotype:
        sum_dev += (distance_route(route) - avg) ** 2
    return sqrt(sum_dev / len(phenotype))


class Individual:
    genotype: List[int]
    code: List[int]
    phenotype: List[List[Client]]
    fitness: float

    def __init__(self, routes: List[List[Client]]):
        self.genotype, self.code = self.encode_solution(routes)
        self.phenotype = routes
        self.fitness = self.get_fitness(func=std_dev)

    def __lt__(self, other) -> bool:
        return self.fitness < other.fitness

    def __repr__(self) -> str:
        return str(self.genotype)

    def encode_solution(
        self, routes: List[List[Client]]
    ) -> tuple[List[int], List[int]]:
        genotype: List[Client] = []
        code: List[int] = []
        for route in routes:
            code.append(len(route) - 1)
            genotype.extend(route[1:])
        return genotype, code

    def decode_solution(self) -> List[List[Client]]:
        phenotype: List[List[int]] = []
        j: int = 0
        for c in self.code:
            route: List[int] = [Client("a", 0.0, 0.0, 0)]
            for i in range(c):
                route.append(self.genotype[j])
                j += 1
            phenotype.append(route)
        return phenotype

    def get_fitness(self, func: Callable[List[List[Client]], float] = average) -> float:
        return func(self.phenotype)


class Child(Individual):
    def __init__(self, genotype: List[int]):
        self.genotype = genotype

    def build_code(self, truck_capacity: int) -> List[int]:
        sum_capacity: int = 0
        code: List[int] = []
        num: int = 0
        for client in self.genotype:
            sum_capacity += client.demand
            if sum_capacity > truck_capacity:
                sum_capacity = client.demand
                code.append(num)
                num = 1
            else:
                num += 1
        code.append(num)
        return code

    def update_child(self, truck_capacity: int) -> None:
        self.code = self.build_code(truck_capacity)
        self.phenotype = self.decode_solution()
        self.fitness = self.get_fitness(func=std_dev)


def gen_first_pop(truck_capacity: int, population_size: int) -> List[Individual]:
    population: List[Individual] = []
    offset = np.linspace(0, 2 * np.pi, population_size)
    thresh_1 = round(INIT_POP_FIRST_THRESHOLD * population_size)
    thresh_2 = round(INIT_POP_SECOND_THRESHOLD * population_size)
    for i in range(population_size):
        routes = sort_clients(
            read_clients(offset[i]),
            truck_capacity,
            reverse=i > thresh_1 and i < thresh_2,
            random=i >= thresh_2,
        )
        population.append(Individual(routes))
    return population


def select_parents(population: List[Individual]) -> List[Individual]:
    parents: List[Individual] = []
    for i in range(len(population) - 2):
        sample: List[Individual] = rd.sample(population, 2)
        parents.append(sample[0] if sample[0] < sample[1] else sample[1])
    parents.extend(sorted(population, key=attrgetter("fitness"), reverse=True)[:2])
    return parents


def order_crossover(parents: List[Individual], truck_capacity: int) -> List[Individual]:
    children: List[Individual] = []
    rd.shuffle(parents)
    it: Iterator[Individual] = iter(parents)
    for parent in it:
        p1: Individual = parent
        p2: Individual = next(it)
        geno_len = len(p1.genotype)
        mid: int = geno_len // 2
        point1 = rd.randint(1, mid)
        point2 = rd.randint(mid, geno_len - 1)
        c1: Child = Child(p1.genotype[point1:point2])
        c2: Child = Child(p2.genotype[point1:point2])
        p2_temp: List[Client] = []
        p1_temp: List[Client] = []
        for i in range(geno_len):
            idx: int = (point2 + i) % geno_len
            if p2.genotype[idx] not in c1.genotype:
                p2_temp.append(p2.genotype[idx])
            if p1.genotype[idx] not in c2.genotype:
                p1_temp.append(p1.genotype[idx])
        for i in range(geno_len - len(c1.genotype)):
            idx: int = (point2 + i) % geno_len
            c1.genotype.insert(idx, p2_temp.pop(0))
            c2.genotype.insert(idx, p1_temp.pop(0))
        c1.update_child(truck_capacity)
        c2.update_child(truck_capacity)
        children.extend([c1, c2])
    return children


def linear_crossover(parents: List[Individual], truck_capacity: int) -> List[Individual]:
    children: List[Individual] = []
    rd.shuffle(parents)
    it: Iterator[Individual] = iter(parents)
    for parent in it:
        p1: Individual = parent
        p2: Individual = next(it)
        geno_len = len(p1.genotype)
        mid: int = geno_len // 2
        point1 = rd.randint(1, mid)
        point2 = rd.randint(mid, geno_len - 1)
        c1: Child = Child(p1.genotype[point1:point2])
        c2: Child = Child(p2.genotype[point1:point2])
        p2_temp: List[Client] = [c for c in p2.genotype if c not in c1.genotype]
        p1_temp: List[Client] = [c for c in p1.genotype if c not in c2.genotype]
        for i in range(geno_len + 1):
            if i < point1 or i > point2:
                c1.genotype.insert(i, p2_temp.pop(0))
                c2.genotype.insert(i, p1_temp.pop(0))
        c1.update_child(truck_capacity)
        c2.update_child(truck_capacity)
        children.extend([c1, c2])
    return children


def position_based_crossover(
    parents: List[Individual], truck_capacity: int
) -> List[Individual]:
    children: List[Individual] = []
    rd.shuffle(parents)
    it: Iterator[Individual] = iter(parents)
    for parent in it:
        p1: Individual = parent
        p2: Individual = next(it)
        geno_len = len(p1.genotype)
        points: List[int] = sorted(
            rd.sample([x for x in range(geno_len)], geno_len // 2)
        )
        p1_temp: List[Client] = [p1.genotype[x] for x in points]
        p2_temp: List[Client] = [x for x in p2.genotype if x not in p1_temp]
        c1: Child = Child([])
        for i in range(geno_len):
            c1.genotype.insert(i, p1_temp.pop(0) if i in points else p2_temp.pop(0))
        p2_temp: List[Client] = [p2.genotype[x] for x in points]
        p1_temp: List[Client] = [x for x in p1.genotype if x not in p2_temp]
        c2: Child = Child([])
        for i in range(geno_len):
            c2.genotype.insert(i, p2_temp.pop(0) if i in points else p1_temp.pop(0))
        c1.update_child(truck_capacity)
        c2.update_child(truck_capacity)
        children.extend([c1, c2])
    return children


def heuristic_crossover(parents: List[Individual], truck_capacity: int) -> List[Individual]:
    children: List[Individual] = []
    for n in range(2):
        parents_temp: List[Individual] = rd.sample(parents, len(parents))
        it: Iterator[Individual] = iter(parents_temp)
        for parent in it:
            p1: Individual = parent
            p2: Individual = next(it)
            geno_len = len(p1.genotype)
            point: int = rd.randint(0, geno_len - 1)
            c: Child = Child([])
            current_client: Client = p1.genotype[point]
            idx_swap: int = p2.genotype.index(current_client)
            p2.genotype[idx_swap], p2.genotype[point] = (
                p2.genotype[point],
                p2.genotype[idx_swap],
            )
            c.genotype.append(current_client)
            for i in range(geno_len - 1):
                idx: int = (point + i) % geno_len
                idx_1: int = (idx + 1) % geno_len
                next_client_1: Client = p1.genotype[idx_1]
                next_client_2: Client = p2.genotype[idx_1]
                if distance_clients(next_client_1, current_client) < distance_clients(
                    next_client_2, current_client
                ):
                    idx_swap = p2.genotype.index(next_client_1)
                    p2.genotype[idx_swap], p2.genotype[idx_1] = (
                        p2.genotype[idx_1],
                        p2.genotype[idx_swap],
                    )
                    current_client = next_client_1
                else:
                    idx_swap: int = p1.genotype.index(next_client_2)
                    p1.genotype[idx_swap], p1.genotype[idx_1] = (
                        p1.genotype[idx_1],
                        p1.genotype[idx_swap],
                    )
                    current_client = next_client_2
                c.genotype.append(current_client)
            c.update_child(truck_capacity)
            children.append(c)
    return children


def edge_recombination_crossover(
    parents: List[Individual], truck_capacity: int
) -> List[Individual]:
    children: List[Individual] = []
    for n in range(2):
        parents_temp: List[Individual] = rd.sample(parents, len(parents))
        it: Iterator[Individual] = iter(parents_temp)
        for parent in it:
            p1: Individual = parent
            p2: Individual = next(it)
            geno_len = len(p1.genotype)
            p1_dict: dict = {}
            p2_dict: dict = {}
            for i in range(geno_len):
                p1_dict[p1.genotype[i]] = [
                    p1.genotype[(i - 1) % geno_len],
                    p1.genotype[(i + 1) % geno_len],
                ]
                p2_dict[p2.genotype[i]] = [
                    p2.genotype[(i - 1) % geno_len],
                    p2.genotype[(i + 1) % geno_len],
                ]
            adj_list: dict = {}
            for key in p1_dict.keys():
                adj_list[key] = p1_dict[key]
                adj_list[key].extend(p2_dict[key])
                adj_list[key] = list(set(adj_list[key]))
            c: Child = Child([])
            current_client: Client = rd.choice([p1.genotype[0], p2.genotype[0]])
            c.genotype.append(current_client)
            for i in range(geno_len - 1):
                for key in adj_list.keys():
                    if current_client in adj_list[key]:
                        adj_list[key].remove(current_client)
                previous_client: Child = current_client
                if len(adj_list[current_client]) > 1:
                    links: List[int] = [len(adj_list[l]) for l in adj_list[current_client]]
                    if len(set(links)) == 1:
                        current_client = rd.choice(adj_list[current_client])
                    else:
                        min_value: int = min(links)
                        min_values: List[int] = [
                            i for i, x in enumerate(links) if x == min_value
                        ]
                        if len(min_values) > 1:
                            current_client = adj_list[current_client][
                                rd.choice(min_values)
                            ]
                        else:
                            current_client = adj_list[current_client][
                                links.index(min_value)
                            ]
                elif len(adj_list[current_client]) == 1:
                    current_client = adj_list[current_client][0]
                else:
                    current_client = rd.choice(
                        [x for x in adj_list.keys() if x != current_client]
                    )
                del adj_list[previous_client]
                c.genotype.append(current_client)
            c.update_child(truck_capacity)
            children.append(c)
    return children


def trajectory_crossover(
    parents: List[Individual], truck_capacity: int
) -> List[Individual]:
    children: List[Individual] = []
    for n in range(2):
        parents_temp: List[Individual] = rd.sample(parents, len(parents))
        it: Iterator[Individual] = iter(parents_temp)
        for parent in it:
            p1: Individual = parent
            p2: Individual = next(it)
            geno_len = len(p1.genotype)
            point: int = rd.randint(0, geno_len - 1)
            c1: Child = cp.deepcopy(p1)
            c2: Child = cp.deepcopy(p2)
            for i in range(geno_len):
                p1_client: Client = p1.genotype[point]
                p2_client: Client = p2.genotype[point]
                if p1_client != p2_client:
                    idx_2: int = c1.genotype.index(p2_client)
                    idx_1: int = c2.genotype.index(p1_client)
                    c1.genotype[point], c1.genotype[idx_2] = (
                        c1.genotype[idx_2],
                        c1.genotype[point],
                    )
                    c2.genotype[point], c2.genotype[idx_2] = (
                        c2.genotype[idx_2],
                        c2.genotype[point],
                    )
                    p2 = c1 if c2 < c1 else c2
                point = (point + i) % geno_len
            children.append(c1)
    return children


def uniform_crossover(parents: List[Individual], truck_capacity: int) -> List[Individual]:
    children: List[Individual] = []
    rd.shuffle(parents)
    it: Iterator[Individual] = iter(parents)
    for parent in it:
        p1: Individual = parent
        p2: Individual = next(it)
        geno_len = len(p1.genotype)
        mask: list[int] = [rd.choice([0, 1]) for x in range(geno_len)]
        c1: Child = Child([p1.genotype[i] for i, x in enumerate(mask) if x == 0])
        c2: Child = Child([p2.genotype[i] for i, x in enumerate(mask) if x == 1])
        for i in range(geno_len):
            if mask[i] == 0:
                c2.genotype.insert(
                    i, [x for x in p1.genotype if x not in c2.genotype][0]
                )
            else:
                c1.genotype.insert(
                    i, [x for x in p2.genotype if x not in c1.genotype][0]
                )
        c1.update_child(truck_capacity)
        c2.update_child(truck_capacity)
        children.extend([c1, c2])
    return children


def order_based_crossover(
    parents: List[Individual], truck_capacity: int
) -> List[Individual]:
    children: List[Individual] = []
    for n in range(2):
        parents_temp: List[Individual] = rd.sample(parents, len(parents))
        it: Iterator[Individual] = iter(parents_temp)
        for parent in it:
            p1: Individual = parent
            p2: Individual = next(it)
            geno_len = len(p1.genotype)
            points: List[int] = sorted(
                rd.sample([x for x in range(geno_len)], geno_len // 2)
            )
            p1_temp: List[Client] = [
                x for x in p1.genotype if p1.genotype.index(x) not in points
            ]
            p2_temp: List[Client] = [x for x in p2.genotype if x not in p1_temp]
            c1: Child = Child([])
            for i in range(geno_len):
                c1.genotype.insert(
                    i,
                    p2_temp.pop(0)
                    if p1.genotype.index(p2.genotype[i]) in points
                    else p1_temp.pop(0),
                )
            c1.update_child(truck_capacity)
            children.append(c1)
    return children


def invertion(mutation_rate: float, children: List[Individual]) -> List[Individual]:
    for child in children:
        mutation = rd.random()
        if mutation <= mutation_rate:
            geno_len = len(child.genotype)
            mid: int = geno_len // 2
            point1 = rd.randint(1, mid)
            point2 = rd.randint(mid, geno_len - 1)
            child.genotype[point1:point2] = child.genotype[point1:point2][::-1]
    return children


def print_pop_stats(population: List[Individual]) -> float:
    fitness: List[float] = []
    for ind in population:
        fitness.append(ind.fitness)
    min_fitness: float = min(fitness)
    # print(f"Score minimal: {min_fitness} ({fitness.count(min_fitness)})\n")
    return min_fitness


def run(
    truck_capacity: int = DEFAULT_TRUCK_CAPACITY,
    population_size: int = DEFAULT_POP_SIZE,
    generation_number: int = DEFAULT_GENERATION_NUMBER,
    cross_function: Callable[[List[Individual]], List[Individual]] = order_crossover,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    mutation_function: Callable[[int, List[Individual]], List[Individual]] = invertion,
) -> tuple[List[Individual], dict]:
    first_population: List[Individual] = gen_first_pop(truck_capacity, population_size)
    population: List[Individual] = first_population
    stats: dict = {}
    for i in range(generation_number):
        # print(f"Génération #{i+1}:")
        stats[i + 1] = print_pop_stats(population)
        parents: List[Individual] = select_parents(population)
        children: List[Individual] = cross_function(parents, truck_capacity)
        population = mutation_function(mutation_rate, children)
    return population, stats
