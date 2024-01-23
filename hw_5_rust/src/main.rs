use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use std::{fs::File, io::Read, thread, time::Duration};

use ndarray::{s, stack, Array, Array1, Array2, ArrayView1, Axis};

const STYLE_TEMPLATE: &str =
    "{prefix:3}[{elapsed_precise}] {msg:20} [{bar:40.cyan/blue}] {pos:>7}/{len:7} {eta}";

fn read_uai_file(filename: &str) -> Result<(Vec<Array1<usize>>, Vec<Array2<f64>>)> {
    let mut file = match File::open(filename) {
        Err(_) => return Err(anyhow::anyhow!("file not found")),
        Ok(file) => file,
    };
    let mut contents = String::new();
    if file.read_to_string(&mut contents).is_err() {
        return Err(anyhow::anyhow!("file read error"));
    }

    let mut lines = contents.lines();

    let _graph_type = lines.next().expect("graph_type not found");
    let _num_vars = lines
        .next()
        .expect("num_vars not found")
        .trim()
        .parse::<usize>()
        .expect("num_vars is not a number");

    let var_degree = lines
        .next()
        .expect("var_degree not found")
        .split_whitespace()
        .map(|x| {
            x.parse::<usize>()
                .expect(format!("{} is not a number", x).as_str())
        })
        .collect::<Vec<usize>>();

    let num_functions = lines
        .next()
        .expect("num_functions not found")
        .trim()
        .parse::<usize>()
        .expect("num_functions is not a number");
    let mut cpt_vars: Vec<Array1<usize>> = vec![Array::zeros((0,)); num_functions];
    let mut original_order: Vec<usize> = Vec::with_capacity(num_functions);
    for _ in 0..num_functions {
        let mut fn_line = lines.next().expect("function not found").split_whitespace();
        let _num_inputs = fn_line
            .next()
            .expect("num_inputs not found")
            .parse::<usize>()
            .expect("num_inputs is not a number");
        let var_vec: Vec<usize> = fn_line
            .map(|x| {
                x.parse::<usize>()
                    .expect(format!("{} is not a number", x).as_str())
            })
            .collect();
        let last_var = var_vec[var_vec.len() - 1];
        original_order.push(last_var);
        let v = Array::from_vec(var_vec);
        cpt_vars[last_var] = v;
    }
    lines.next().expect("empty line not found");
    let mut cpt_tables = vec![Array::zeros((0, 0)); num_functions];
    for i in 0..num_functions {
        let num_values = lines
            .next()
            .expect("num_values not found")
            .trim()
            .parse::<usize>()
            .expect("num_values is not a number");
        // table is array with shape [num_values/degree, degree]
        let mut table = Array::zeros((num_values / var_degree[i], var_degree[i]));
        let mut index = 0;
        while let Some(line) = lines.next() {
            if line.trim().is_empty() {
                break;
            }
            let row = line.split_whitespace().map(|x| {
                x.parse::<f64>()
                    .expect(format!("{} is not a number", x).as_str())
            }).collect::<Vec<f64>>();
            if row.len() == var_degree[i] {
                table.row_mut(index).assign(&Array::from_vec(row));
            } else {
                for (j, x) in row.iter().enumerate() {
                    let col = j % var_degree[i];
                    table[[index, col]] = *x;
                    if col == var_degree[i] - 1 {
                        index += 1;
                    }
                }
            }
            index += 1;
        }
        cpt_tables[original_order[i]] = table;
    }
    println!("parsing uai file");
    return Ok((cpt_vars, cpt_tables));
}

fn load_training_data(training_file: &str) -> Array2<usize> {
    let mut file = File::open(training_file).expect("file not found");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("file read error");
    let mut lines = contents.lines();
    // l1 num_vars num_samples
    let mut l1_iter = lines.next().expect("l1 not found").split_whitespace();
    let num_vars = l1_iter
        .next()
        .expect("num_vars not found")
        .parse::<usize>()
        .expect("num_vars is not a number");
    let num_samples = l1_iter
        .next()
        .expect("num_samples not found")
        .parse::<usize>()
        .expect("num_samples is not a number");
    let mut data = Array::zeros((num_samples, num_vars));
    for (i, line) in lines.enumerate() {
        for (j, x) in line.split_whitespace().enumerate() {
            // if x is ? set usize::MAX
            data[[i, j]] = x.parse::<usize>().unwrap_or(usize::MAX);
        }
    }
    return data;
}

fn maximum_likelihood_estimation(
    cpt_vars: Vec<Array1<usize>>,
    training_file: &str,
) -> Vec<Array2<f64>> {
    let data = load_training_data(training_file);
    let mut cpt_tables = Vec::with_capacity(cpt_vars.len());
    let style = indicatif::ProgressStyle::default_bar()
        .template(STYLE_TEMPLATE)
        .unwrap()
        .progress_chars("#>-");
    let bar = ProgressBar::new(cpt_vars.len() as u64);
    bar.set_message("learning cpt tables");
    bar.set_prefix("── ");
    bar.set_style(style.clone());
    for (i, cpt_var) in cpt_vars.iter().enumerate() {
        if cpt_var.len() == 1 {
            // numerator = number of items in ith column of data equal to 0
            let numerator = 1 + data.column(i).iter().filter(|x| **x == 0).count();
            let theta = numerator as f64 / (2.0 + data.nrows() as f64);
            // add [[theta, 1-theta]] to cpt_tables
            cpt_tables.push(Array::from_shape_vec((1, 2), vec![theta, 1.0 - theta]).unwrap());
        } else {
            let num_parents = cpt_var.len() - 1;
            let var_index = cpt_var[num_parents];
            // assignments is a 2^num_parents x num_parents array with each row being the binary representation of the row number
            let assignments =
                Array::from_shape_fn((2usize.pow(num_parents as u32), num_parents), |(r, c)| {
                    (r >> (num_parents - c - 1)) & 1
                });
            // parent_values is a num_samples x num_parents array with each row being the values of the parents of the ith sample
            let parent_values = data.select(Axis(1), &cpt_var.slice(s![..-1]).to_slice().unwrap());

            // denominator is a 1 x 2^num_parents array with each element being the number of samples where the values of the parents are equal to the corresponding row of assignments
            let denominator = Array::from_shape_fn((2usize.pow(num_parents as u32),), |r| {
                parent_values
                    .rows()
                    .into_iter()
                    .filter(|x| x == assignments.row(r))
                    .count() as f64
                    + 2.0
            });
            // numerator is a 1 x 2^num_parents array
            //  with each element being the number of samples where the values of the parents are equal to the corresponding row of assignments and the value of the child is 0
            let numerator = Array::from_shape_fn((2usize.pow(num_parents as u32),), |r| {
                parent_values
                    .rows()
                    .into_iter()
                    .zip(data.column(var_index).iter())
                    .filter(|(x, y)| x == assignments.row(r) && **y == 0)
                    .count() as f64
                    + 1.0
            });
            let theta = numerator / denominator;
            let cpt_table = stack(Axis(1), &[theta.view(), theta.mapv(|x| 1.0 - x).view()])
                .expect("stack error");
            cpt_tables.push(cpt_table);
        }
        bar.inc(1);
    }
    bar.finish();
    return cpt_tables;
}

fn log_likelihood_diff(
    cpt_vars: Vec<Array1<usize>>,
    cpt_tables: Vec<Array2<f64>>,
    learned_tables: Vec<Array2<f64>>,
    test_file: &str,
) -> f64 {
    let style = indicatif::ProgressStyle::default_bar()
        .template(STYLE_TEMPLATE)
        .unwrap()
        .progress_chars("#>-");
    let bar = ProgressBar::new(0);
    bar.set_prefix("── ");
    bar.set_style(style.clone());
    let diff = _lld_bar(cpt_vars, cpt_tables, learned_tables, test_file, bar);
    return diff;
}


fn _lld_bar(cpt_vars: Vec<Array1<usize>>, cpt_tables: Vec<Array2<f64>>, learned_tables: Vec<Array2<f64>>, test_file: &str, bar: ProgressBar) -> f64 {
    let data = load_training_data(test_file);
    bar.set_length(data.nrows() as u64);
    bar.set_message("calculating ll diff");
    let mut ll_real = Array::zeros((data.nrows(),));
    let mut ll_learned = Array::zeros((data.nrows(),));
    for (i, row) in data.rows().into_iter().enumerate() {
        ll_learned[i] = probability_of_assignment(&cpt_vars, &learned_tables, &row);
        ll_real[i] = probability_of_assignment(&cpt_vars, &cpt_tables, &row);
        bar.inc(1);
    }
    bar.finish_with_message("finished ll diff");
    let diff: f64 = ll_real
        .iter()
        .zip(ll_learned.iter())
        .map(|(x, y)| (x.ln() - y.ln()).abs())
        .sum();
    return diff;
}

fn probability_of_assignment(
    cpt_vars: &Vec<Array1<usize>>,
    cpt_tables: &Vec<Array2<f64>>,
    assignment: &ArrayView1<usize>,
) -> f64 {
    let mut prob = 1.0;
    for cpt_var in cpt_vars {
        let parent_indexes = cpt_var.slice(s![..-1]);
        let var = cpt_var[cpt_var.len() - 1];
        let parent_values = assignment.select(Axis(0), &parent_indexes.to_slice().unwrap());
        let index = parent_values
            .iter()
            .rev()
            .enumerate()
            .fold(0, |acc, (i, x)| acc + (x << i));
        let conditional_prob = cpt_tables[var][[index, assignment[var]]];
        prob *= conditional_prob;
    }
    return prob;
}

fn gen_random_cpt_tables(cpt_vars: &Vec<Array1<usize>>) -> Vec<Array2<f64>> {
    let mut cpt_tables = Vec::with_capacity(cpt_vars.len());
    for cpt_var in cpt_vars {
        let num_parents = cpt_var.len() - 1;
        let half_table = Array::random((2usize.pow(num_parents as u32),), Uniform::new(0.0, 1.0));
        let table = stack(
            Axis(1),
            &[half_table.view(), half_table.mapv(|x| 1.0 - x).view()],
        )
        .expect("stack error");
        cpt_tables.push(table);
    }
    return cpt_tables;
}

fn expecttation_maximization(
    cpt_vars: Vec<Array1<usize>>,
    training_file: &str,
    iterations: usize,
    bar: ProgressBar,
) -> Vec<Array2<f64>> {
    let data = load_training_data(training_file);
    bar.set_message("learning cpt tables");
    let mut cpt_tables = gen_random_cpt_tables(&cpt_vars);
    for _ in 0..iterations {
        let (data_weights, filled_data) = fill_missing_data(&cpt_vars, &cpt_tables, &data);
        cpt_tables.clear();
        for var in &cpt_vars {
            let num_parents = var.len() - 1;
            let var_index = var[var.len() - 1];
            if var.len() == 1 {
                let numerator = filled_data
                    .column(var_index)
                    .iter()
                    .zip(data_weights.iter())
                    .filter(|(x, _)| **x == 0)
                    .map(|(_, y)| y)
                    .sum::<f64>()
                    + 1.0;
                let denominator = 2.0 + data_weights.sum();
                let theta = numerator / denominator;
                let cpt_table = Array::from_shape_vec((1, 2), vec![theta, 1.0 - theta]).unwrap();
                cpt_tables.push(cpt_table);
                continue;
            }

            let assignments =
                Array::from_shape_fn((2usize.pow(num_parents as u32), num_parents), |(r, c)| {
                    (r >> (num_parents - c - 1)) & 1
                });
            let values = filled_data.select(Axis(1), var.as_slice().unwrap());
            let denominator = Array::from_shape_fn((2usize.pow(num_parents as u32),), |r| {
                values
                    .rows()
                    .into_iter()
                    .zip(data_weights.iter())
                    .filter(|(x, _)| x.slice(s![..-1]) == assignments.row(r))
                    .map(|(_, y)| y)
                    .sum::<f64>()
                    + 2.0
            });
            let numerator = Array::from_shape_fn((2usize.pow(num_parents as u32),), |r| {
                values
                    .rows()
                    .into_iter()
                    .zip(data_weights.iter())
                    .filter(|(x, _)| x.slice(s![..-1]) == assignments.row(r) && x[x.len() - 1] == 0)
                    .map(|(_, y)| y)
                    .sum::<f64>()
                    + 1.0
            });
            let theta = numerator / denominator;
            let cpt_table = stack(Axis(1), &[theta.view(), theta.mapv(|x| 1.0 - x).view()])
                .expect("stack error");
            cpt_tables.push(cpt_table);
        }
        bar.inc(1);
    }
    bar.finish_with_message("finished learning cpt tables");
    return cpt_tables;
}

fn fill_missing_data(
    cpt_vars: &Vec<Array1<usize>>,
    cpt_tables: &Vec<Array2<f64>>,
    data: &Array2<usize>,
) -> (Array1<f64>, Array2<usize>) {
    let num_missing_per_row = data
        .rows()
        .into_iter()
        .map(|x| x.iter().filter(|y| **y == usize::MAX).count())
        .collect::<Vec<usize>>();
    let missing_size = num_missing_per_row
        .iter()
        .map(|x| 2usize.pow(*x as u32))
        .sum::<usize>();

    let mut filled_data = Array::zeros((missing_size, data.ncols()));
    let mut data_weights = Array::zeros((missing_size,));

    let mut j = 0;
    for (i, row) in data.rows().into_iter().enumerate() {
        if num_missing_per_row[i] == 0 {
            filled_data
                .row_mut(i)
                .assign(&Array::from_iter(row.iter().map(|x| *x)));
            data_weights[i] = 1.0;
            j += 1;
            continue;
        }

        let added_rows = 2usize.pow(num_missing_per_row[i] as u32);
        let assignments = Array::from_shape_fn((added_rows, num_missing_per_row[i]), |(r, c)| {
            (r >> (num_missing_per_row[i] - c - 1)) & 1
        });
        let missing_indexes = row
            .iter()
            .enumerate()
            .filter(|(_, x)| **x == usize::MAX)
            .map(|(i, _)| i)
            .collect::<Vec<usize>>();
        // make added_rows copies of row
        let mut new_rows = Array::from_shape_fn((added_rows, row.len()), |(_, c)| {
            return row[c];
        });
        for (a, &col) in missing_indexes.iter().enumerate() {
            new_rows
                .slice_mut(s![.., col])
                .assign(&assignments.slice(s![.., a]));
        }
        let probs = Array::from_shape_fn((added_rows,), |r| {
            probability_of_assignment(&cpt_vars, &cpt_tables, &new_rows.row(r))
        });
        let sum = probs.sum();
        let weights = probs / sum;
        filled_data
            .slice_mut(s![j..j + added_rows, ..])
            .assign(&new_rows);
        data_weights
            .slice_mut(s![j..j + added_rows])
            .assign(&weights);
        j += added_rows;
    }
    return (data_weights, filled_data);
}

fn run_em(cpt_vars: Vec<Array1<usize>>, cpt_tables: Vec<Array2<f64>>, training_file: &str, test_file: &str) -> (f64, f64) {
    // run 5 expextation maximization with 5 different random initializations
    // and find the mean and stdev of the log likelihood difference
    // run in parallel
    let mut diffs = Vec::with_capacity(5);
    let mut threads = Vec::with_capacity(5);
    let (tx, rx) = std::sync::mpsc::channel();

    let multi_bar = MultiProgress::new();

    for id in 0..5 {
        let tx = tx.clone();
        let cpt_vars = cpt_vars.clone();
        let cpt_tables = cpt_tables.clone();
        let training_file = training_file.to_owned().clone();
        let test_file = test_file.to_owned().clone();
        let style = indicatif::ProgressStyle::default_bar()
            .template(STYLE_TEMPLATE)
            .unwrap()
            .progress_chars("#>-");
        let em_bar = multi_bar.add(ProgressBar::new(20));
        em_bar.set_prefix(format!("[{}/5] ", id + 1));
        em_bar.set_style(style.clone());
        let t = thread::spawn(move || {
            let learned_tables = expecttation_maximization(cpt_vars.clone(), training_file.as_str(), 20, em_bar.clone());
            let diff = _lld_bar(
                cpt_vars,
                cpt_tables,
                learned_tables,
                test_file.as_str(),
                em_bar,
            );
            tx.send(diff).unwrap();
        });
        threads.push(t);
    }

    for t in threads {
        let diff = rx.recv().unwrap();
        diffs.push(diff);
        t.join().unwrap();
    }

    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let stdev = (diffs
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / diffs.len() as f64)
        .sqrt();
    return (mean, stdev);

}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        args.push("2".to_owned());
        args.push("f".to_owned());
        args.push("1".to_owned());
    }
    let dataset = args[1].parse::<usize>().expect("dataset is not a number");
    let mode = &args[2];
    let train_file = args[3].parse::<usize>().expect("train_file is not a number");

    let uaifile = format!("hw5-data/dataset{}/{}.uai", dataset, dataset);
    let trainfile = format!(
        "hw5-data/dataset{}/train-{}-{}.txt",
        dataset, mode, train_file
    );
    let testfile = format!("hw5-data/dataset{}/test.txt", dataset);

    let (cpt_vars, cpt_tables) = read_uai_file(&uaifile).expect("read uai file error");

    let start = std::time::Instant::now();
    if mode == "f" {
        let learned_tables = maximum_likelihood_estimation(cpt_vars.clone(), &trainfile);
        let diff = log_likelihood_diff(cpt_vars, cpt_tables, learned_tables, &testfile);
        println!("diff: {:.3}", diff);
    } else if mode == "p" {
        let (diff, stdev) = run_em(cpt_vars.clone(), cpt_tables.clone(), &trainfile, &testfile);
        println!("diff: {:.3} +/- {:.3}", diff, stdev);
    } else {
        println!("mode not found");
    }
    let duration = start.elapsed();
    let duration = Duration::from_secs(duration.as_secs());
    println!("Finished in {}", humantime::format_duration(duration));
}
