#!/usr/bin/env python3

import argparse
import subprocess
import re, os, sys
from tqdm import tqdm
from collections import defaultdict

def extract_smallest_loss(output):
    print(output)
    val_losses = list(map(float, re.findall(r'Validation Loss: ([0-9]+\.[0-9]+)', output)))
    aucs = list(map(float, re.findall(r'AUC: ([0-9]+\.[0-9]+)', output)))
    print('Losses:', val_losses, 'AUCs:', aucs)
    return min(val_losses, default=1), max(aucs, default=0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', default='cce', help='Method')
    parser.add_argument('-e', '--epochs', default=10, help='Epochs')
    parser.add_argument('-d', '--dataset', default='ml-100k', help='Dataset')
    parser.add_argument('-b', '--batch-size', default=256, type=int, help='Batch Size')
    parser.add_argument('-lo', '--lo-pow', default=1, type=int)
    parser.add_argument('-hi', '--hi-pow', default=12, type=int)


    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    lo_pow = args.lo_pow
    hi_pow = args.hi_pow

    losses = defaultdict(list)
    aucs = defaultdict(list)

    arg_prod = [(2**i, 2**j) for i in range(lo_pow, hi_pow+1) for j in range(0, i+1)]
    for ppd, n_chunks in tqdm(arg_prod, desc="Overall Progress"):
        cmd = [
            sys.executable,
            os.path.join(script_dir, 'movielens.py'),
            '--method', args.method,
            '--ppd', str(ppd),
            '--n-chunks', str(n_chunks),
            '--dataset', args.dataset,
            '--batch-size', str(args.batch_size),
            '--epochs', str(args.epochs),
        ]

        print(f'Running {ppd=} {n_chunks=}')
        output = subprocess.check_output(cmd).decode('utf-8')
        smallest_loss, top_auc = extract_smallest_loss(output)
        losses[ppd].append(smallest_loss)
        aucs[ppd].append(top_auc)

    # Print & write results
    # Print to file and output at the same time
    def write_to_file_and_print(file, text):
        print(text, file=file)
        print(text)

    for typ, vals in [('ll', losses), ('auc', aucs)]:
        file_name = f'chunks.{args.dataset}.{args.method}.{typ}'
        print('Writing results to', file_name)
        with open(file_name, 'a') as file:
            write_to_file_and_print(file, f"## {args.method}")

            header = "ppd"
            for j in range(0, hi_pow+1):
                header += f"\tn-chunks={2**j}"
            write_to_file_and_print(file, header)

            for i in range(lo_pow, hi_pow + 1):
                ppd = 2**i
                line = [str(ppd)] + list(map(str, vals[ppd]))
                write_to_file_and_print(file, "\t".join(line))
            file.flush()

