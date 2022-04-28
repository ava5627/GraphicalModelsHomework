import os
import re
import statistics
import pandas as pd


def extract_info(filename):

    lze_pattern = re.compile("log10\\(Z\\) = ([\\d.]*)")
    z_hat = []
    with open(filename) as output_file:
        for line in output_file:
            match = re.search(lze_pattern, line)
            if match:
                z_hat += [float(match.group(1))]
    return z_hat


def zPR(filename):
    with open('./networks/' + filename + '.PR') as file:
        file.readline()
        return float(file.readline())


def process_output(print_lines):

    data = pd.DataFrame(columns=['File', 'w', 't', 'q', 'value'])
    for filename in sorted(os.listdir('./output/')):
        grid, w, t, q = filename.split('.')
        z_real = zPR(filename[:8] + '.uai')
        print_str = f"{filename[:8]} | {w=:2}{t=:4}{q=:2} | **error** | "
        output_file = './output/' + filename
        z_hat = extract_info(output_file)

        error = []
        for z_hat in z_hat:
            error += [abs((z_real - z_hat) / z_real)]

        mean = statistics.mean(error)
        stdev = statistics.stdev(error)
        value = f'{mean:1.3e}+/-{stdev:1.3e}'
        data.loc[len(data)] = [filename[:8], int(w), int(t), int(q), value]
        print_str += f' {mean:1.3e}+/-{stdev:1.3e} |'
        if print_lines:
            print(print_str)
    return data


def data_stuff(data: pd.DataFrame):
    fwm = data.groupby(['File', 'w']).sum().index
    qtm = data.groupby(['q', 't']).sum().index

    table = pd.DataFrame(columns=qtm, index=fwm)

    for f in data['File'].unique():
        for w in data['w'].unique():
            vs = data[(data['File'] == f) & (data['w'] == w)].sort_values(['q', 't'])['value']
            table.loc[f, w] = vs.tolist()
    print(table)


def run_all():
    dir = './networks/'
    for filename in os.listdir(dir):
        if not filename.endswith('.uai'):
            continue

        f = dir + filename
        for w in [1, 3, 7]:

            command = f"python3 w_cutset.py {f} {w} {f}.order {f}.{w}cutset"
            os.system(command)
            for t in [10, 20, 100]:
                for q in [1, 5]:
                    output_file = f"./output/{filename[:-4]}.{w}.{t}.{q}"
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    print(f'running {filename[:-4]}: {w=}{t=}{q=}')
                    for i in range(5):
                        command_wcis = f"./wcis_code/wcis {f} {f}.order {f}.{w}cutset {t} {q} >> {output_file}"
                        os.system(command_wcis)


def main():
    run_all()
    data = process_output(print_lines=False)
    data_stuff(data)


if __name__ == "__main__":
    main()
