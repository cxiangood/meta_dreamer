#!/usr/bin/env python3
"""Fetch training logs from HPC and plot curves locally."""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Collect from stdin or local file
def load_logs(path):
    data = {"sigreg": [], "kl": []}
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            tag = "sigreg" if "sigreg_loss" in d.get("wm", {}) or any("sigreg" in k for k in d.keys() if k.startswith("wm/")) else "kl"
            # Detect from metrics
            for k in d.keys():
                if "sigreg_loss" in k:
                    data["sigreg"].append(d)
                    break
                elif "kl_loss" in k:
                    data["kl"].append(d)
                    break
    return data

def main():
    if len(sys.argv) < 2:
        # Try fetching from HPC first
        import os
        import subprocess
        logdir = os.path.expanduser("~/metadrive/meta_dreamer_pytorch/logs")
        os.makedirs(logdir, exist_ok=True)

        print("Fetching logs from HPC...")
        for tag in ["sigreg_p2", "kl_p2"]:
            remote = f"hpcAmd:/share/home/u23516/code/meta_dreamer-main/pytorch/logs/{tag}/training_log.jsonl"
            local = os.path.join(logdir, f"{tag}_training_log.jsonl")
            subprocess.run(["scp", "-o", "ConnectTimeout=10", remote, local], check=False)
            print(f"  {tag}: {'OK' if os.path.exists(local) else 'FAILED'}")

        paths = [
            os.path.join(logdir, "sigreg_p2_training_log.jsonl"),
            os.path.join(logdir, "kl_p2_training_log.jsonl"),
        ]
    else:
        paths = sys.argv[1:]

    # Load all logs
    all_data = []
    for path in paths:
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        all_data.append(data)
        tag = "SIGReg" if "sigreg" in path else "KL" if "kl" in path else path
        print(f"{tag}: {len(data)} entries")

    # Build per-model curves
    sigreg_steps, sigreg_rec, sigreg_dyn, sigreg_rew, sigreg_con, sigreg_reg = [], [], [], [], [], []
    kl_steps, kl_rec, kl_dyn, kl_rew, kl_con, kl_reg = [], [], [], [], [], []

    for data in all_data:
        for d in data:
            step = d["global_step"]
            wm = {k.replace("wm/", ""): v for k, v in d.items() if k.startswith("wm/")}

            if "sigreg_loss" in wm:
                sigreg_steps.append(step)
                sigreg_rec.append(wm.get("loss_rec", 0))
                sigreg_dyn.append(wm.get("loss_dyn", 0))
                sigreg_rew.append(wm.get("loss_rew", 0))
                sigreg_con.append(wm.get("loss_con", 0))
                sigreg_reg.append(wm["sigreg_loss"])
            elif "kl_loss" in wm:
                kl_steps.append(step)
                kl_rec.append(wm.get("loss_rec", 0))
                kl_dyn.append(wm.get("loss_dyn", 0))
                kl_rew.append(wm.get("loss_rew", 0))
                kl_con.append(wm.get("loss_con", 0))
                kl_reg.append(wm["kl_loss"])

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("SIGReg vs KL Baseline — Phase 2 Training Curves", fontsize=14, fontweight="bold")

    def plot_pair(ax, title, sig_x, sig_y, kl_x, kl_y, ylabel):
        ax.plot(sig_x, sig_y, 'b-', label='SIGReg', alpha=0.8, linewidth=1.5)
        ax.plot(kl_x, kl_y, 'r-', label='DreamerV3(KL)', alpha=0.8, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plot_pair(axes[0, 0], "Regularization Loss\n(lower KL=1.0 = collapse!)",
              sigreg_steps, sigreg_reg, kl_steps, kl_reg, "Loss")
    plot_pair(axes[0, 1], "Reconstruction Loss (MSE)",
              sigreg_steps, sigreg_rec, kl_steps, kl_rec, "MSE")
    plot_pair(axes[0, 2], "Total Dynamics Loss",
              sigreg_steps, sigreg_dyn, kl_steps, kl_dyn, "Loss")
    plot_pair(axes[1, 0], "Reward Prediction Loss",
              sigreg_steps, sigreg_rew, kl_steps, kl_rew, "MSE")
    plot_pair(axes[1, 1], "Continue Prediction Loss",
              sigreg_steps, sigreg_con, kl_steps, kl_con, "BCE")
    plot_pair(axes[1, 2], "Total Loss",
              sigreg_steps, [r+d+w+c for r,d,w,c in zip(sigreg_rec, sigreg_dyn, sigreg_rew, sigreg_con)],
              kl_steps, [r+d+w+c for r,d,w,c in zip(kl_rec, kl_dyn, kl_rew, kl_con)],
              "Loss")

    plt.tight_layout()
    out_path = "/tmp/training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
