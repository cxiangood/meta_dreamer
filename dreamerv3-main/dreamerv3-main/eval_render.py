#!/usr/bin/env python3
"""
eval_render.py

Run a short single-environment evaluation with rendering enabled.
Saves frames to an output folder and optionally composes an MP4.

Usage:
  python3 eval_render.py --steps 300 --outdir ./render_out --save_video

This is intended to be run inside WSL2 with WSLg (Windows 11) so the realtime
Panda/MetaDrive window appears on your desktop. If WSLg is not available the
script will still save frames to disk.
"""
import argparse
import os
import time
import numpy as np
import threading

try:
    from embodied.envs.metadrive_lane_keeping import MetaDriveLaneKeeping
except Exception as e:
    raise SystemExit(f'Failed importing MetaDriveLaneKeeping: {e}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=300, help='Number of env steps')
    p.add_argument('--outdir', type=str, default='./render_out', help='Where to save frames/video')
    p.add_argument('--render_interval', type=int, default=1, help='Save every N frames')
    p.add_argument('--width', type=int, default=640, help='Render width')
    p.add_argument('--height', type=int, default=480, help='Render height')
    p.add_argument('--steering', type=float, default=0.0, help='Constant steering value')
    p.add_argument('--throttle', type=float, default=0.5, help='Constant throttle/brake value')
    p.add_argument('--save_frames', action='store_true', help='Save individual frames as PNG')
    p.add_argument('--save_video', action='store_true', help='Compose an mp4 from saved frames (requires imageio)')
    p.add_argument('--follow_log', type=str, default='', help='Path to training log to follow and print DREAMER-ACTION lines')
    p.add_argument('--manual', action='store_true', help='Enable manual control mode in the MetaDrive env (if supported)')
    return p.parse_args()


def main():
    args = parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f'Creating MetaDrive env with render ON ({args.width}x{args.height})')
    env = MetaDriveLaneKeeping('lane_keeping', size=(args.width, args.height), use_render=True, manual_control=bool(args.manual))

    # If requested, spawn a background thread to follow the training log and
    # print any DREAMER-ACTION lines as they appear. This lets you run eval
    # visualization while also seeing policy action outputs emitted by the
    # training process (from train_dreamerv3_metadrive.sh).
    stop_follow = threading.Event()
    def follow_log(path):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                # Seek to end
                f.seek(0, os.SEEK_END)
                while not stop_follow.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(0.2)
                        continue
                    if 'DREAMER-ACTION' in line:
                        print(line.rstrip())
        except FileNotFoundError:
            print(f'Follow log file not found: {path}')
        except Exception as e:
            print(f'Error following log {path}: {e}')

    if args.follow_log:
        t = threading.Thread(target=follow_log, args=(args.follow_log,), daemon=True)
        t.start()

    # Reset
    if hasattr(env, '_reset'):
        obs = env._reset()
    else:
        # Some wrappers expect a reset action
        obs = env.step({'reset': True, 'steering': 0.0, 'throttle_brake': 0.0})

    saved = []
    t0 = time.time()
    for step in range(args.steps):
        action = {'steering': float(args.steering), 'throttle_brake': float(args.throttle), 'reset': False}
        try:
            obs = env.step(action)
        except Exception as e:
            print(f'Env.step() raised: {e}')
            break

        # Try to get an RGB frame. Prefer env.render() (Panda window) or obs['image']
        frame = None
        try:
            frame = env.render()
        except Exception:
            frame = None

        if frame is None and isinstance(obs, dict) and 'image' in obs:
            frame = obs['image']

        if frame is not None:
            # If user requested saving frames, save periodically.
            if isinstance(frame, np.ndarray) and args.save_frames and (step % args.render_interval == 0):
                fn = os.path.join(outdir, f'frame_{step:06d}.png')
                try:
                    # Save via PIL to avoid extra deps
                    from PIL import Image
                    im = Image.fromarray(frame.astype('uint8'))
                    im.save(fn)
                    saved.append(fn)
                    print(f'Saved frame {fn}')
                except Exception as e:
                    print(f'Failed to save frame: {e}')
        # Sleep a tiny bit so the Panda window can update (if it exists)
        time.sleep(0.01)

    duration = time.time() - t0
    print(f'Ran {step+1} steps in {duration:.1f}s, saved {len(saved)} frames to {outdir}')

    if args.save_video and len(saved) > 0:
        try:
            import imageio
            video_path = os.path.join(outdir, 'render.mp4')
            print(f'Composing video {video_path}...')
            writer = imageio.get_writer(video_path, fps=25)
            for fn in sorted(saved):
                img = imageio.v2.imread(fn)
                writer.append_data(img)
            writer.close()
            print('Wrote', video_path)
        except Exception as e:
            print('Failed to write video (imageio missing or error):', e)

    try:
        env.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
