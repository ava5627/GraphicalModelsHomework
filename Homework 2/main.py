import sys
import math
import random
import statistics
import time
from copy import deepcopy


def fill_size(network, node):
    f = 0
    children = network[node]
    for i in children:
        f += len([c for c in children if c not in network[i] and c != i])
    return f


def min_fill(network):
    d = math.inf
    n = []

    for k in network:
        k_fill = fill_size(network, k)
        if k_fill < d:
            n, d = [k], k_fill
        if k_fill == d:
            n += [k]
    return n[random.randint(0, len(n) - 1)]


def min_degree(network):
    d = math.inf
    n = []

    for k in network:
        if len(network[k]) < d:
            n, d = [k], len(network[k])
        if len(network[k]) == d:
            n += [k]
    return n[random.randint(0, len(n) - 1)]


def main():
    if len(sys.argv) != 5:
        sys.exit("Invalid number of arguments")

    _, input_file, method, num_runs, output_file = sys.argv

    print(f"Getting {method} order for {input_file}")

    with open(input_file) as network_file:
        network_file.readline()

        num_nodes = int(network_file.readline())
        static_network = {i: [] for i in range(num_nodes)}

        network_file.readline()
        network_file.readline()

        for line in network_file:
            if line.strip() == '':
                break
            _, *clique = line.split()
            for i in clique:
                for j in clique:
                    if i != j:
                        static_network[int(i)] += [int(j)]

        heuristic = min_degree if method == 'min_degree' else min_fill

        best_order = []
        widths = [0] * int(num_runs)
        best_width = math.inf

        start = time.perf_counter()
        for i in range(int(num_runs)):
            network = deepcopy(static_network)
            order = []
            width = 0
            while network:
                next_node = heuristic(network)
                order += [next_node]
                width = max(width, len(network[next_node]))
                children = network.pop(next_node)
                for j in children:
                    network[j].remove(next_node)
                    network[j] += [c for c in children if c not in network[j] and c != j]
            widths[i] = width
            if width < best_width:
                best_width = widths[i]
                best_order = order

        end = time.perf_counter()
        print(best_order)
        print(f"{best_width=}")
        print(f"Mean width over {num_runs} runs: {statistics.mean(widths)}")
        print(f"Standard Deviation over {num_runs} runs: {statistics.stdev(widths)}")
        print(f"Average time per ordering: {(end - start) / int(num_runs)}")
        # print(f"| {method: <10}{input_file: <33} | {best_width: 2} | {statistics.mean(widths): 1.4f} | {statistics.stdev(widths): 1.4f} | {(end - start) / int(num_runs): 1.4f} |")
        with open(output_file, 'w') as output:
            output.write(' '.join(map(str, best_order)))


if __name__ == "__main__":
    main()
