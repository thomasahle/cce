#!/usr/bin/env python3

import argparse
import subprocess
import re, os, sys
from tqdm import tqdm

def extract_smallest_loss(output):
    print(output)
    val_losses = list(map(float, re.findall(r'Validation Loss: ([0-9]+\.[0-9]+)', output)))
    aucs = list(map(float, re.findall(r'AUC: ([0-9]+\.[0-9]+)', output)))
    print('Losses:', val_losses, 'AUCs:', aucs)
    return min(val_losses, default=1), max(aucs, default=0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', default='cce', help='Method')
    parser.add_argument('-e', '--epochs', default='', help='Epochs')
    parser.add_argument('-d', '--dataset', default='ml-100k', help='Dataset')
    parser.add_argument('-w', '--workers', default='', help='Workers')
    parser.add_argument('-b', '--batch-size', default='256', help='Batch Size')
    parser.add_argument('-l', '--lo-pow', default=1, type=int)
    parser.add_argument('-hi', '--hi-pow', default=12, type=int)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    runs = 3
    lo_pow = args.lo_pow
    hi_pow = args.hi_pow

    ppds = []
    losses = []
    aucs = []

    arg_prod = [(seed, 2**i) for seed in range(1, runs+1) for i in range(lo_pow, hi_pow+1)]
    for seed, ppd in tqdm(arg_prod, desc="Overall Progress"):
        cmd = [
            sys.executable,
            os.path.join(script_dir, 'movielens.py'),
            '--method', args.method,
            '--ppd', str(ppd),
            '--seed', str(seed),
            '--dataset', args.dataset,
            '--batch-size', args.batch_size
        ]
        if args.epochs:
            cmd.extend(['--epochs', args.epochs])
        if args.workers:
            cmd.extend(['--num-workers', args.workers])

        print(f'Running {ppd=} {seed=}')
        output = subprocess.check_output(cmd).decode('utf-8')
        smallest_loss, top_auc = extract_smallest_loss(output)
        ppds.append(ppd)
        losses.append(smallest_loss)
        aucs.append(top_auc)

    # Print & write results
    # Print to file and output at the same time
    def write_to_file_and_print(file, text):
        print(text, file=file)
        print(text)

    for typ, vals in [('ll', losses), ('auc', aucs)]:
        file_name = f'results.{args.dataset}.{args.method}.{typ}'
        print('Writing results to', file_name)
        with open(file_name, 'a') as file:
            write_to_file_and_print(file, f"## {args.method}")

            header = "ppd"
            for run in range(1, runs + 1):
                header += f"\tseed_{run}"
            write_to_file_and_print(file, header)

            for i in range(lo_pow, hi_pow + 1):
                line = str(ppds[i - lo_pow])
                for run in range(1, runs + 1):
                    index = (run - 1) * (hi_pow - lo_pow + 1) + (i - lo_pow)
                    line += f"\t{vals[index]}"
                write_to_file_and_print(file, line)
            file.flush()
