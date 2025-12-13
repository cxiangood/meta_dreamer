#!/usr/bin/env python3
"""
Example script showing how to use the CARLA lane keeping environment with DreamerV3.

Prerequisites:
1. Install CARLA 0.10.0 from https://carla.readthedocs.io/en/0.10.0/getting_started/
2. Start CARLA server: ./CarlaUnreal.sh (Linux) or CarlaUnreal.exe (Windows)
3. Install required dependencies:
   pip install opencv-python

Usage:
# Train DreamerV3 on CARLA lane keeping task
python dreamerv3/main.py --configs carla --logdir ~/logdir/carla/{timestamp}

# Debug mode (smaller networks, faster training)
python dreamerv3/main.py --configs carla debug --logdir ~/logdir/carla/{timestamp}

# Custom configuration
python dreamerv3/main.py \
  --configs carla0915 \
  --env.carla.port 2000 \
  --env.carla.town Town02 \
  --env.carla.weather CloudyNoon \
  --run.steps 500000 \
  --logdir ~/logdir/carla0915/{timestamp}
"""

import subprocess
import sys
import time
import argparse

def check_carla_server(host='localhost', port=2000):
    """Check if CARLA server is running"""
    try:
        import carla
        client = carla.Client(host, port)
        client.set_timeout(2.0)
        client.get_server_version()
        return True
    except Exception as e:
        print(f"CARLA server not accessible at {host}:{port}: {e}")
        return False

def start_training(config_name='carla', logdir=None, extra_args=None):
    """Start DreamerV3 training on CARLA"""
    
    # Check if CARLA server is running
    if not check_carla_server():
        print("Error: CARLA server is not running!")
        print("Please start CARLA server first:")
        print("  Linux: ./CarlaUE4.sh")
        print("  Windows: CarlaUE4.exe")
        print("  Or with custom settings: ./CarlaUE4.sh -opengl -quality-level=Low")
        return False
    
    print("CARLA server detected. Starting training...")
    
    # Build command
    cmd = [
        sys.executable, 
        '/home/xiongxi/桌面/worldmodel_dreamerv3/dreamer/dreamerv3/main.py',
        '--configs', config_name
    ]
    
    if logdir:
        cmd.extend(['--logdir', logdir])
    else:
        cmd.extend(['--logdir', '~/logdir/carla0915/{timestamp}'])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Train DreamerV3 on CARLA lane keeping task')
    parser.add_argument('--config', default='carla0915', help='Config name (default: carla0915)')
    parser.add_argument('--debug', action='store_true', help='Use debug config for faster iteration')
    parser.add_argument('--logdir', help='Log directory')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--town', default='Town01', help='CARLA town (Town01-Town10)')
    parser.add_argument('--weather', default='ClearNoon', 
                       choices=['ClearNoon', 'CloudyNoon', 'WetNoon', 'ClearSunset'],
                       help='Weather preset')
    parser.add_argument('--steps', type=int, help='Number of training steps')
    
    args = parser.parse_args()
    
    # Build config name
    config_name = args.config
    if args.debug:
        config_name += ' debug'
    
    # Build extra arguments
    extra_args = []
    
    # 强制使用CPU后端避免JAX CUDA问题
    extra_args.extend(['--jax.platform', 'cpu'])
    extra_args.extend(['--jax.prealloc', 'False'])
    
    if args.port != 2000:
        extra_args.extend(['--env.carla.port', str(args.port)])
    if args.town != 'Town01':
        extra_args.extend(['--env.carla.town', args.town])
    if args.weather != 'ClearNoon':
        extra_args.extend(['--env.carla.weather', args.weather])
    if args.steps:
        extra_args.extend(['--run.steps', str(args.steps)])
    
    # Start training
    success = start_training(config_name, args.logdir, extra_args)
    
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
