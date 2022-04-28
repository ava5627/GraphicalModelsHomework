import sys
import numpy as np
import time
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor


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


def print_table(cpt_vars, cpt_tables):
    for f, v in zip(cpt_vars, cpt_tables):
        print(f)
        for i in v:
            print(f"\t{i}")


def maximum_liklihood_estimation(cpt_vars, training_file):
    data = np.loadtxt(training_file, skiprows=1, dtype=int)

    cpt_tables = []

    for *parents, var in cpt_vars:
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


def probability_of_assignment(cpt_vars, cpt_tables, assignment):
    prob = 1
    power_list = 2**np.arange(16)[::-1]
    for (*parents, var), table in zip(cpt_vars, cpt_tables):
        parent_values = assignment[parents]

        # convert binary list to decimal index
        table_index = parent_values.dot(power_list[16-len(parents):])

        cond_prob = table[table_index][assignment[var]]
        prob *= cond_prob
    return prob


def log_likelihood_diff(cpt_vars, cpt_tables, learned_tables, test_file):
    data = np.loadtxt(test_file, skiprows=1, dtype=int)
    ll_real = np.zeros(data.shape[0])
    ll_learned = np.zeros(data.shape[0])
    avg_time = 0
    for i in range(data.shape[0]):
        start = time.time()
        if i % 1000 == 0:
            print(f"\rLL DIFF: {i:6}/{data.shape[0]}"
                  f"                "
                  f"{i/data.shape[0]:7.2%} "
                  f"ETC: {timedelta(seconds=int(avg_time*(data.shape[0]-i)))}", end="")
        ll_real[i] = probability_of_assignment(cpt_vars, cpt_tables, data[i])
        ll_learned[i] = probability_of_assignment(cpt_vars, learned_tables, data[i])
        avg_time = (avg_time * i + time.time() - start) / (i+1)
    print()
    diff = np.abs(np.log(ll_real) - np.log(ll_learned)).sum()
    return diff


def fill_data(data, cpt_vars, cpt_tables, msg, etc):
    missing_values = (data == -1)

    missing_size = np.exp2(np.count_nonzero(missing_values, axis=1)).sum(dtype=int)
    filled_data = np.zeros((missing_size, data.shape[1]))
    data_weights = np.zeros(missing_size)

    start = time.time()
    j = 0
    for i, row in enumerate(data):
        percent = j/missing_size/2
        print(f"{msg} {percent:7.2%} ETC: {etc - timedelta(seconds=int(time.time() - start - 2))}", end="")

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
    avg_time = data.shape[0] / 100
    for i in range(iterations):
        start = time.time()
        msg = f"\rEM run {run_id} |{'#'*(i + 1)}{' '*(iterations-i-1)}| {i+1:02}/{iterations:2}"
        percent = 0
        etc = timedelta(seconds=int(avg_time * (iterations-i)))
        print(f"{msg} {percent:7.2%} ETC: {etc}", end="\x1b[K")

        # Fill in missing values and weight using the current cpt_tables
        data_weights, filled_data = fill_data(data, cpt_vars, cpt_tables, msg, etc)
        del cpt_tables
        cpt_tables = []

        j = 0
        for *parents, var in cpt_vars:
            percent = .5 + j/len(cpt_vars)/2
            j += 1
            print(f"{msg} {percent:7.2%} ETC: {etc - timedelta(seconds=int(time.time() - start - 2))}", end="")

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
        avg_time = (avg_time * i + time.time() - start) / (i+1)
    print(end=' ')
    return cpt_tables


def run(input_uai_filename, task_id, training_file, test_file, parallel=False):
    parallel = int(parallel)
    cpt_vars, cpt_tables = read_uai_file(input_uai_filename)
    if task_id in ['f', '1']:
        start = time.time()
        learned_tables = maximum_liklihood_estimation(cpt_vars, training_file)
        learning_time = time.time() - start
        print(f'Finished learning in {timedelta(seconds=int(learning_time))}')
        diff = log_likelihood_diff(cpt_vars, cpt_tables, learned_tables, test_file)
        diff_time = time.time() - start - learning_time
        print(f'Finished in {timedelta(seconds=int(diff_time))}')
        print(f'Log-likelihood difference: {diff}')
        return f"Log Likelihood Difference = {diff}"

    elif task_id in ['p', '2'] and not parallel:
        diffs = np.zeros(5)
        for i in range(5):
            diffs[i] = run_em(cpt_vars, cpt_tables, training_file, test_file, i)
        mean = diffs.mean()
        stdev = diffs.std()
        print(f"Log Likelihood Difference: {mean} +/- {stdev}")
        return f"Log Likelihood Difference = {mean} +/- {stdev}"
    elif task_id in ['p', '2'] and parallel:
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(run_em, cpt_vars, cpt_tables, training_file, test_file, i))
            diffs = [future.result() for future in futures]
        mean = np.mean(diffs)
        stdev = np.std(diffs)
        print(f"Log Likelihood Difference: {mean} +/- {stdev}")
        return f"Log Likelihood Difference = {mean} +/- {stdev}"
    else:
        sys.exit(f'Task id must be 1, 2, f, or p not {task_id}')


def run_em(cpt_vars, cpt_tables, training_file, test_file, run_id):
    print(f"Starting run: {run_id}")
    start = time.time()
    learned_tables = expectation_maximization(cpt_vars, training_file, iterations=20, run_id=run_id)
    learning_time = time.time() - start
    print(f'{run_id} finished learning in {timedelta(seconds=int(learning_time))}')
    diff = log_likelihood_diff(cpt_vars, cpt_tables, learned_tables, test_file)
    diff_time = time.time() - start - learning_time
    print(f'{run_id} Finished with in {timedelta(seconds=int(diff_time))}')
    print(f'{run_id} Log-likelihood difference: {diff}')
    return diff


def main():
    if len(sys.argv) not in [5, 6]:
        sys.exit("python3 main.py <input-uai-file> <task-id> <training-data> <test-data> [parallel]")

    _, input_uai_filename, task_id, training_file, test_file, *parallel = sys.argv
    print(run(input_uai_filename, task_id, training_file, test_file, *parallel))


if __name__ == "__main__":
    main()
