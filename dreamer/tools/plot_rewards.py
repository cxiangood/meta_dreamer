#!/usr/bin/env python3
"""Plot per-episode rewards from CSV produced by eval_metadrive.py or the env logger."""
import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_rewards(path):
    xs = []
    ys = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                ys.append(float(row.get('total_reward', 0.0)))
                xs.append(i)
            except Exception:
                continue
    return np.array(xs), np.array(ys)


def smooth(y, window=5):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=os.path.expanduser('~/metadrive_eval_rewards.csv'))
    parser.add_argument('--smooth', type=int, default=5)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    xs, ys = read_rewards(args.log)
    if len(xs) == 0:
        print('No data found in', args.log)
        return

    plt.figure(figsize=(8,4))
    plt.plot(xs, ys, label='raw', alpha=0.4)
    if args.smooth and args.smooth > 1:
        ys_s = smooth(ys, args.smooth)
        xs_s = xs[:len(ys_s)]
        plt.plot(xs_s, ys_s, label=f'smoothed({args.smooth})', color='C1')
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.title('MetaDrive: reward per episode')
    plt.legend()
    plt.grid(True)
    if args.out:
        plt.savefig(args.out)
        print('Saved plot to', args.out)
    else:
        plt.show()


if __name__ == '__main__':
    main()
