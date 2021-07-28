import math
import random as rd
from operator import attrgetter
from typing import List

from data.data import Client


def sort_clients(
    liste_clients: List[Client],
    capacity: int,
    reverse: bool = False,
    random: bool = False,
) -> List[List[Client]]:
    depot: Client = liste_clients[0]
    liste: List[Client] = (
        sorted(liste_clients, key=attrgetter("angle"), reverse=reverse)
        if not random
        else rd.sample(liste_clients, len(liste_clients))
    )
    liste.remove(depot)
    sum_demand: int = 0
    routes: List[List[CLient]] = [[]]
    route_num: int = 0
    routes[route_num].append(depot)
    for client in liste:
        sum_demand += client.demande
        if sum_demand > capacity:
            route_num += 1
            routes.append([])
            routes[route_num].append(depot)
            sum_demand = client.demand
        routes[route_num].append(client)
    return routes


def distance_clients(client1: Client, client2: Client) -> float:
    return round(math.sqrt((client2.x - client1.x) ** 2 + (client2.y - client1.y) ** 2))


def distance_route(route: List[Client]) -> float:
    distance: int = 0
    for i in range(len(route)):
        i_2 = (i + 1) % len(route)
        distance += distance_clients(route[i_2], route[i])
    return distance
