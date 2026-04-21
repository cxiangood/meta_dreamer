#!/usr/bin/env python3
"""Poll a Slurm job log every N seconds and print a compact status summary."""

import argparse
import pathlib
import subprocess
import time


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def tail(path, n):
    p = pathlib.Path(path)
    if not p.exists():
        return f"[missing] {path}"
    lines = p.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--err", default="")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--lines", type=int, default=30)
    args = parser.parse_args()

    while True:
        print(f"\n===== monitor {time.strftime('%F %T')} job={args.jobid} =====", flush=True)
        print(run(["squeue", "-j", args.jobid]), flush=True)
        print("----- stdout tail -----", flush=True)
        print(tail(args.out, args.lines), flush=True)
        if args.err:
            print("----- stderr tail -----", flush=True)
            print(tail(args.err, args.lines), flush=True)
        status = run(["squeue", "-h", "-j", args.jobid])
        if not status:
            print(f"[Monitor] job {args.jobid} no longer in squeue; stopping.", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
