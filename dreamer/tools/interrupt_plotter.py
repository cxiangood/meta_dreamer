"""Register signal handlers to save reward plot on interrupt.

This module is imported for its side-effect: it registers SIGINT/SIGTERM
handlers that will attempt to read the reward CSV and save a plot, then
exit. It is best imported early in the training entrypoint.
"""
import os
import sys
import signal
import traceback

def _safe_plot(csv_path, out_path):
    try:
        csv_path = os.path.expanduser(csv_path)
        out_path = os.path.abspath(os.path.expanduser(out_path))
        if not os.path.exists(csv_path):
            print(f"[interrupt_plotter] No CSV at {csv_path}, nothing to plot")
            return out_path
        # lazy import of plotting utilities
        try:
            from dreamer.tools import plot_rewards
        except Exception:
            # fallback: simple matplotlib plot
            import csv
            import numpy as np
            ys = []
            with open(csv_path) as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        ys.append(float(row.get('total_reward', 0)))
                    except Exception:
                        pass
            xs = np.arange(len(ys))
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,3))
            plt.plot(xs, ys, alpha=0.6)
            if len(ys) > 1:
                # simple smoothing
                from numpy import convolve, ones
                w = min(5, max(1, len(ys)//4))
                ys_s = convolve(ys, ones(w)/w, mode='valid')
                plt.plot(xs[:len(ys_s)], ys_s, label='smooth')
            plt.xlabel('episode')
            plt.ylabel('total reward')
            plt.grid(True)
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            return out_path

        # use the plot module
        xs, ys = plot_rewards.read_rewards(csv_path)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3))
        plt.plot(xs, ys, alpha=0.6)
        if len(ys) > 1:
            plt.plot(xs[:len(ys)], plot_rewards.smooth(ys, 5), label='smooth')
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.grid(True)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
    except Exception:
        print('[interrupt_plotter] Failed to save plot:', file=sys.stderr)
        traceback.print_exc()
        return None


def _handler(signum, frame):
    try:
        csv_path = os.environ.get('METADRIVE_REWARD_LOG', '~/metadrive_eval_rewards.csv')
        out_path = os.environ.get('METADRIVE_INTERRUPT_PLOT', '~/metadrive_eval_rewards_plot_interrupt.png')
        saved = _safe_plot(csv_path, out_path)
        if saved:
            print(f"[interrupt_plotter] Saved interrupt plot to {saved}")
        else:
            print('[interrupt_plotter] No plot saved')
    except Exception:
        print('[interrupt_plotter] Exception in handler')
        traceback.print_exc()
    # restore default handler and re-raise KeyboardInterrupt to terminate
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # exit with code 130 (130 is common for SIGINT)
    try:
        sys.exit(130)
    except SystemExit:
        raise


# register handlers
try:
    signal.signal(signal.SIGINT, _handler)
except Exception:
    pass
try:
    signal.signal(signal.SIGTERM, _handler)
except Exception:
    pass
