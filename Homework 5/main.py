import sys
import numpy as np
import time
import os
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm, trange


def read_uai_file(filename):
    with open(filename) as uai_file:

        uai_file.readline()  # graph_type
        uai_file.readline()  # num_vars
        uai_file.readline()  # vars

        num_functions = int(uai_file.readline().strip())
        cpt_vars = [[]] * num_functions
        var_order = []
        for line in uai_file:
            if line.strip() == "":
                break
            num_values, *function = list(map(int, line.split()))
            cpt_vars[function[-1]] = function
            var_order.append(function[-1])

        cpt_tables = [[]]*len(cpt_vars)
        index = 0
        for *parents, var in cpt_vars:
            num_values = int(uai_file.readline().strip())
            cpt_values = []
            line = uai_file.readline().strip()
            while line != "":
                values = line.split()
                values = np.array(values, dtype=float).reshape(len(values)//2, 2)
                cpt_values += [values]
                line = uai_file.readline().strip()
            cpt_values = np.array(cpt_values).reshape(num_values//2, 2)
            cpt_tables[var_order[index]] = cpt_values
            index += 1

        return cpt_vars, cpt_tables


def maximum_liklihood_estimation(cpt_vars, training_file):
    data = np.loadtxt(training_file, skiprows=1, dtype=int)

    cpt_tables = []

    for *parents, var in tqdm(cpt_vars, desc="MLE         ", ncols=100, ascii="->=", colour="cyan"):
        cpt_values = []

        if not parents:
            numerator = 1 + np.count_nonzero(data[:, var] == 0)
            theta = numerator / (2 + data.shape[0])
            cpt_values = np.array([[theta, 1 - theta]])
        else:
            # Get list of binary assignments for parents
            # loop over all assignments to parents and count the number of
            # data points that match that assignment
            assignments = np.unpackbits(np.arange(2**len(parents), dtype='>i2').view(np.uint8)).reshape(
                -1, 16)[:, -len(parents):]

            denominator_indices = (data[:, np.newaxis, parents] == assignments).all(axis=2)
            denominator = denominator_indices.sum(axis=0)
            numerator = ((data[:, np.newaxis, var] == 0) & denominator_indices).sum(axis=0)
            thetas = ((1 + numerator) / (2 + denominator)).reshape(-1, 1)
            cpt_values = np.hstack((thetas, 1 - thetas))
        cpt_tables.append(cpt_values)

    return cpt_tables


def gen_random_cpt_table(cpt_vars):
    cpt_tables = []
    for var in cpt_vars:
        cpt_values = []
        for i in range(2**(len(var) - 1)):
            p = np.random.rand()
            cpt_values.append([p, 1-p])
        cpt_tables.append(cpt_values)
    return cpt_tables


def probability_of_assignment(cpt_vars, cpt_tables, assignment: np.ndarray):
    prob = 1
    power_list = 2**np.arange(16)[::-1]
    for (*parents, var), table in zip(cpt_vars, cpt_tables):
        parent_values: np.ndarray = assignment[parents]

        # convert binary list to decimal index
        table_index = parent_values.dot(power_list[16-len(parents):])

        cond_prob = table[table_index][assignment[var]]
        prob *= cond_prob
    return prob


def log_likelihood_diff(cpt_vars, cpt_tables, learned_tables, test_file, run_id=""):
    data = np.loadtxt(test_file, skiprows=1, dtype=int)
    ll_real = np.zeros(data.shape[0])
    ll_learned = np.zeros(data.shape[0])
    bar_position = 2 * (run_id if run_id else 0)
    for i in trange(data.shape[0], desc=f"LL DIFF {run_id:<4}", ncols=100, ascii="->=", colour="cyan", leave=False, position=bar_position, nrows=10):
        ll_real[i] = probability_of_assignment(cpt_vars, cpt_tables, data[i])
        ll_learned[i] = probability_of_assignment(cpt_vars, learned_tables, data[i])
    diff = np.abs(np.log(ll_real) - np.log(ll_learned)).sum()
    return diff


def fill_data(data, cpt_vars, cpt_tables, bar_position=None):
    missing_values = (data == -1)

    missing_size = np.exp2(np.count_nonzero(missing_values, axis=1)).sum(dtype=int)
    filled_data = np.zeros((missing_size, data.shape[1]))
    data_weights = np.zeros(missing_size)

    j = 0
    for i, row in tqdm(
        enumerate(data),
        desc="Fill Data   ",
        ncols=100,
        ascii="->=",
        colour="cyan",
        total=data.shape[0],
        leave=False,
        position=bar_position + 1,
        nrows=10
    ):
        # if no missing values, use the original data and a weight of 1
        if not missing_values.any():
            data_weights = np.append(data_weights, 1)
            filled_data = np.append(filled_data, row, axis=0)
            continue

        new_size = 2**np.count_nonzero(missing_values[i])
        new_rows = np.tile(row, (new_size, 1))

        # generate all possible assignments for the missing values
        # using numpy's unpack bits function
        assignments = np.unpackbits(np.arange(new_size, dtype=np.uint8)).reshape(
            -1, 8)[:, -np.count_nonzero(missing_values[i]):]

        # set the columns of the new rows to the values in the assignments
        new_rows[:, missing_values[i]] = assignments

        # calculate the probability of each row in the new_rows
        # using numpy
        probs = np.zeros(new_size)
        for k, row in enumerate(new_rows):
            probs[k] = probability_of_assignment(cpt_vars, cpt_tables, row)

        # Normalize probs to get weights
        weights = probs/probs.sum()
        data_weights[j:j+new_size] = weights
        filled_data[j:j+new_size] = new_rows
        j += new_size

    return data_weights, filled_data


def expectation_maximization(cpt_vars, training_file, iterations=20, run_id=None):
    np.random.seed(run_id)
    data = np.genfromtxt(training_file, skip_header=1, missing_values='?', filling_values=-1, dtype=int)

    cpt_tables = []

    cpt_tables = gen_random_cpt_table(cpt_vars)
    bar_position = 2 * (run_id if run_id is not None else 0)
    for i in trange(iterations, desc=f"EM {run_id:<9}", ncols=100, ascii="->=", colour="cyan", position=bar_position, leave=False, nrows=10):

        # Fill in missing values and weight using the current cpt_tables
        data_weights, filled_data = fill_data(data, cpt_vars, cpt_tables, bar_position)
        del cpt_tables
        cpt_tables = []

        j = 0
        for *parents, var in tqdm(cpt_vars, desc="EM Iteration", ncols=100, ascii="->=", colour="cyan", leave=False, position=bar_position+1, nrows=10):
            j += 1

            if not parents:
                numerator_indices = (filled_data[:, var] == 0)
                theta = (1 + data_weights[numerator_indices].sum()) / (2 + data_weights.sum())
                cpt_values = np.array([[theta, 1 - theta]])

            else:
                # Get list of binary assignments for parents
                assignments = np.unpackbits(np.arange(2**len(parents), dtype='>i2').view(np.uint8)).reshape(
                    -1, 16)[:, -len(parents):]

                # Get the indices of the rows that meet the assignments
                denominator_indices = (filled_data[:, np.newaxis, parents] == assignments).all(axis=2).T
                numerator_indices = (filled_data[:, var] == 0).T & denominator_indices

                # Calculate weights
                numerator_weights = np.where(numerator_indices, data_weights, 0).sum(axis=1)
                denominator_weights = np.where(denominator_indices, data_weights, 0).sum(axis=1)

                # Calculate probabilities
                thetas = ((1 + numerator_weights) / (2 + denominator_weights)).reshape(-1, 1)
                cpt_values = np.hstack([thetas, 1 - thetas])

            cpt_tables.append(cpt_values)
    return cpt_tables


def run(input_uai_filename, task_id, training_file, test_file, parallel=False):
    parallel = int(parallel)
    cpt_vars, cpt_tables = read_uai_file(input_uai_filename)
    if task_id in ['f', '1']:
        learned_tables = maximum_liklihood_estimation(cpt_vars, training_file)
        diff = log_likelihood_diff(cpt_vars, cpt_tables, learned_tables, test_file)
        return f"Log Likelihood Difference = {diff}"

    elif task_id in ['p', '2'] and not parallel:
        diffs = np.zeros(5)
        for i in range(5):
            diffs[i] = run_em(cpt_vars, cpt_tables, training_file, test_file, i)
        mean = diffs.mean()
        stdev = diffs.std()
        return f"Log Likelihood Difference = {mean} +/- {stdev}"
    elif task_id in ['p', '2'] and parallel:
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(run_em, cpt_vars, cpt_tables, training_file, test_file, i))
            diffs = [future.result() for future in futures]
        mean = np.mean(diffs)
        stdev = np.std(diffs)
        return f"Log Likelihood Difference = {mean} +/- {stdev}"
    else:
        sys.exit(f'Task id must be 1, 2, f, or p not {task_id}')


def run_em(cpt_vars, cpt_tables, training_file, test_file, run_id):
    learned_tables = expectation_maximization(cpt_vars, training_file, iterations=20, run_id=run_id)
    diff = log_likelihood_diff(cpt_vars, cpt_tables, learned_tables, test_file, run_id=run_id)
    return diff


def main():
    if len(sys.argv) < 4 or (sys.argv[2] == 'p' and len(sys.argv) != 5):
        sys.exit("python3 main.py dataset mode file parallel")

    _, dataset, task_id, training_file, *parallel = sys.argv
    input_uai_filename = f"./hw5-data/dataset{dataset}/{dataset}.uai"
    training_file = f"./hw5-data/dataset{dataset}/train-{task_id}-{training_file}.txt"
    test_file = f"./hw5-data/dataset{dataset}/test.txt"
    if not os.path.exists(input_uai_filename):
        sys.exit(f"Dataset {dataset} does not exist")
    if not os.path.exists(training_file):
        sys.exit(f"Training file {training_file} does not exist")
    if not os.path.exists(test_file):
        sys.exit(f"Test file {test_file} does not exist")
    start = time.time()
    print("\n"*10, run(input_uai_filename, task_id, training_file, test_file, *parallel))
    print(f"Finished in {timedelta(seconds=int(time.time() - start))}")


if __name__ == "__main__":
    main()
