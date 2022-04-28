import sys
import random
from copy import deepcopy


def make_clusters(network, order):

    clusters = {i: {i} for i in order}

    processed = []
    for i in order:
        children = network[i]
        processed += [i]
        for j in children:
            for c in children:
                if c not in processed and j not in processed:
                    network[c].add(j)
                    network[j].add(c)

    processed = []
    for i in order:
        clusters[i] = set(j for j in network[i] if j not in processed)
        processed += [i]

    return clusters


def count_clusters(vars, clusters):
    counts = {}
    for i in vars:
        counts[i] = 0
        for c in clusters:
            if i in clusters[c]:
                counts[i] += 1
    return counts


def max_number_of_clusters(counts):
    max_list = []
    max_count = 0
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for i, c in sorted_counts:
        if c > max_count:
            max_list = [i]
            max_count = c
        elif c == max_count:
            max_list += [i]
    random.shuffle(max_list)
    return max_list[0]


def get_max_cluster_size(clusters):
    max_cluster_size = 0
    for c in clusters:
        max_cluster_size = max(max_cluster_size, len(clusters[c]))
    return max_cluster_size


def w_cutset(uai_filename, w, order_filename, output_filename):
    with open(uai_filename) as network_file:
        network_file.readline()

        num_nodes = int(network_file.readline())
        static_network = {i: set() for i in range(num_nodes)}

        network_file.readline()
        network_file.readline()

        for line in network_file:
            if line.strip() == '':
                break
            _, *clique = line.split()
            for i in clique:
                for j in clique:
                    static_network[int(i)].add(int(j))

    with open(order_filename) as order_file:
        order_str = order_file.readline().split()
        order = list(map(int, order_str))

    network = deepcopy(static_network)
    clusters = make_clusters(network, order)
    counts = count_clusters(order, clusters)

    cut_set = set()

    while counts and get_max_cluster_size(clusters) > w:
        max_node = max_number_of_clusters(counts)
        counts.pop(max_node)
        cut_set.add(max_node)
        for c in clusters:
            if max_node in clusters[c]:
                clusters[c].remove(max_node)

    with open(output_filename, 'w') as output_file:
        output_file.write(f'{len(cut_set)} ')
        for c in cut_set:
            output_file.write(f'{c} ')


def main():
    if len(sys.argv) != 5:
        print('usage: uai file, w, order file, output file')
        exit()
    _, uai_file, w, order_file, output_file = sys.argv
    w = int(w)

    w_cutset(uai_file, w, order_file, output_file)


if __name__ == "__main__":
    main()
