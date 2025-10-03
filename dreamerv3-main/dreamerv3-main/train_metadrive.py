#!/usr/bin/env python3
"""
Launch script for training DreamerV3 on MetaDrive Lane Keeping task
"""

import os
import sys
import subprocess

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dreamerv3_dir = os.path.join(script_dir, "dreamerv3")
    
    # Set up environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = script_dir + os.pathsep + env.get('PYTHONPATH', '')
    
    # Command to run DreamerV3 with MetaDrive
    cmd = [
        sys.executable, 
        os.path.join(dreamerv3_dir, "main.py"),
        "--configs", "metadrive_lane_keeping"
    ]
    
    print("Starting DreamerV3 training on MetaDrive Lane Keeping...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {script_dir}")
    print("-" * 60)
    
    try:
        # Run the command
        subprocess.run(cmd, cwd=script_dir, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()