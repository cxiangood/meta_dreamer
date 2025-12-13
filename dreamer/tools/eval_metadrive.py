#!/usr/bin/env python3
"""Run deterministic MetaDrive rollouts and record per-episode rewards.

Usage: python dreamer/tools/eval_metadrive.py --episodes 50 --seed 42
"""
import os
import sys
import time
import argparse
import csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from embodied.envs import metadrive_lane_keeping as md_env_mod


def run_eval(episodes, seed, log_path, render=False, snapshot_out=None, plot_on_interrupt=False):
    # ensure fixed seed env var for the environment
    if seed is not None:
        os.environ['METADRIVE_FIXED_SEED'] = str(seed)
    if log_path is None:
        log_path = os.path.expanduser('~/metadrive_eval_rewards.csv')

    env = md_env_mod.MetaDriveLaneKeeping(task=None, size=(64, 64))

    # ensure CSV header exists
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('episode,total_reward,length,seed,reason\n')

    # attempt to import plotting helper for optional snapshots
    plot_mod = None
    try:
        from dreamer.tools import plot_rewards as plot_mod
    except Exception:
        plot_mod = None

    # prepare bookkeeping variables so KeyboardInterrupt handler can reference them
    in_episode = False
    total = 0.0
    length = 0
    try:
        for ep in range(episodes):
            # trigger reset
            obs = env.step({'reset': True, 'steering': 0.0, 'throttle_brake': 0.0})
            total = 0.0
            length = 0
            done = False
            in_episode = True
            while not obs.get('is_last', False):
                # simple fixed policy: small throttle forward
                action = {'reset': False, 'steering': 0.0, 'throttle_brake': 0.3}
                obs = env.step(action)
                total += float(obs.get('reward', 0.0))
                length += 1
                if render:
                    try:
                        img = env.render()
                    except Exception:
                        pass

            # collect final reward if provided
            seed_used = os.environ.get('METADRIVE_FIXED_SEED', '')
            reason = 'UNK'
            try:
                reason = 'CRASH' if obs.get('is_terminal', False) else ('DONE' if obs.get('is_last', False) else 'OTHER')
            except Exception:
                pass

            with open(log_path, 'a') as f:
                f.write(f"{ep},{total:.6f},{length},{seed_used},{reason}\n")

            print(f"Episode {ep:3d}: reward={total:.3f} len={length} seed={seed_used}")

            # small pause to allow logs to flush
            time.sleep(0.01)

            # optional: save snapshot plot after each episode
            if snapshot_out and plot_mod is not None:
                try:
                    # use the plot module to save a snapshot image
                    # plot_mod expects CLI; call its functions directly
                    xs, ys = plot_mod.read_rewards(log_path)
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(6,3))
                    plt.plot(xs, ys, alpha=0.6)
                    if len(ys) >= 1:
                        plt.plot(xs[:len(ys)], plot_mod.smooth(ys, max(1, min(5, len(ys)//4))), label='smooth')
                    plt.xlabel('episode')
                    plt.ylabel('total reward')
                    plt.title('rewards')
                    plt.grid(True)
                    plt.tight_layout()
                    # expand and make absolute path
                    snap_path = os.path.abspath(os.path.expanduser(snapshot_out))
                    os.makedirs(os.path.dirname(snap_path) or '.', exist_ok=True)
                    plt.savefig(snap_path)
                    plt.close()
                    print('Saved snapshot to', snap_path)
                except Exception as e:
                    print('Snapshot save failed:', e)
    except KeyboardInterrupt:
        print('\n[eval_metadrive] Caught KeyboardInterrupt — flushing current partial episode if any...')
        # if an episode was in progress, write partial result
        try:
            if in_episode and (length > 0):
                ep_idx = None
                try:
                    # try to infer current episode index from file (count lines)
                    with open(log_path, 'r') as f:
                        # header line included, so subtract header
                        ep_idx = sum(1 for _ in f) - 1
                except Exception:
                    ep_idx = None
                seed_used = os.environ.get('METADRIVE_FIXED_SEED', '')
                reason = 'INTERRUPT_PARTIAL'
                try:
                    with open(log_path, 'a') as f:
                        f.write(f"{ep_idx if ep_idx is not None else -1},{total:.6f},{length},{seed_used},{reason}\n")
                    print(f"Wrote partial episode record: reward={total:.3f} len={length}")
                except Exception as e:
                    print('Failed to append partial episode record:', e)
        except Exception as e:
            print('Failed to write partial episode:', e)

        # optionally generate a plot on interrupt
        if plot_on_interrupt and plot_mod is not None:
            try:
                out = snapshot_out or os.path.expanduser('~/metadrive_eval_snapshot.png')
                out = os.path.abspath(os.path.expanduser(out))
                xs, ys = plot_mod.read_rewards(log_path)
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6,3))
                plt.plot(xs, ys, alpha=0.6)
                if len(ys) >= 1:
                    plt.plot(xs[:len(ys)], plot_mod.smooth(ys, max(1, min(5, len(ys)//4))), label='smooth')
                plt.xlabel('episode')
                plt.ylabel('total reward')
                plt.title('rewards')
                plt.grid(True)
                plt.tight_layout()
                os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
                plt.savefig(out)
                plt.close()
                print('Saved interrupt snapshot to', out)
            except Exception as e:
                print('Plot-on-interrupt failed:', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log', type=str, default=os.path.expanduser('~/metadrive_eval_rewards.csv'))
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--snapshot', type=str, default=None, help='Write a snapshot plot to this path after each episode')
    parser.add_argument('--plot-on-interrupt', action='store_true', help='Generate a plot if execution is interrupted')
    args = parser.parse_args()
    run_eval(args.episodes, args.seed, args.log, args.render, snapshot_out=args.snapshot, plot_on_interrupt=args.plot_on_interrupt)


if __name__ == '__main__':
    main()
