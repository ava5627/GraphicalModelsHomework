from main import run
import os


def already_run(file, dataset, mode,):
    if not os.path.exists(f"./output_{dataset}.txt"):
        return False
    with open(f"./output_{dataset}.txt", 'r') as output_file:
        lines = output_file.readlines()
        for line in lines:
            if f"Dataset {dataset}, File: {mode}{file}" in line:
                return True
    return False


def main():
    for dataset in [1, 2, 3]:
        for mode in ['f', 'p']:
            for file in range(1, 5):
                if already_run(file, dataset, mode):
                    continue
                with open(f"./output_{dataset}.txt", 'a+') as output_file:
                    print(f"{dataset}-{mode}-{file}")
                    model_file = f"./hw5-data/dataset{dataset}/{dataset}.uai"
                    train_file = f"./hw5-data/dataset{dataset}/train-{mode}-{file}.txt"
                    test_file = f"./hw5-data/dataset{dataset}/test.txt"
                    output = run(model_file, mode, train_file, test_file, parallel=1)
                    output_file.write(f"Dataset {dataset}, File: {mode}{file} - {output}\n")


if __name__ == "__main__":
    main()
