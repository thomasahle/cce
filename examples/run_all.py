#!/usr/bin/env python3

import argparse
import subprocess
import re, os, sys
from tqdm import tqdm

def extract_smallest_loss(output):
    print(output)
    val_losses = re.findall(r'Validation Loss: ([0-9]+\.[0-9]+)', output)
    val_losses = [float(loss) for loss in val_losses]
    print('Losses:', val_losses)
    return min(val_losses) if val_losses else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', default='cce', help='Method')
    parser.add_argument('-e', '--epochs', default='', help='Epochs')
    parser.add_argument('-d', '--dataset', default='ml-100k', help='Dataset')
    parser.add_argument('-w', '--workers', default='', help='Workers')
    parser.add_argument('-b', '--batch-size', default='256', help='Batch Size')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    runs = 3
    lo_pow = 1
    hi_pow = 12

    ppds = []
    losses = []

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
        smallest_loss = extract_smallest_loss(output)
        ppds.append(ppd)
        losses.append(smallest_loss)

    # Print results
    header = "ppd"
    for seed in range(1, runs + 1):
        header += f"\tseed_{seed}"
    print(header)
    
    for i in range(lo_pow, hi_pow + 1):
        line = str(ppds[i - lo_pow])
        for run in range(1, runs + 1):
            index = (run - 1) * (hi_pow - lo_pow + 1) + (i - lo_pow)
            line += f"\t{losses[index]}"
        print(line)
