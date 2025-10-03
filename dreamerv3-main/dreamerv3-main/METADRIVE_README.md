# MetaDrive Lane Keeping Integration

This project integrates MetaDrive lane keeping environment with reinforcement learning frameworks, supporting both DreamerV3 (Linux/WSL2) and Stable-Baselines3 (Windows native).

## Features

- **Visual Observations**: 64x64 RGB camera images from the vehicle's perspective
- **Kinematic States**: Speed, acceleration, angular velocity (roll, pitch, yaw rates)
- **Control Inputs**: Steering angle and throttle/brake values
- **Navigation Info**: Distance to route center and route completion progress
- **Realistic Physics**: MetaDrive's sophisticated vehicle dynamics and traffic simulation

## Installation

### For Windows Users (Recommended)

1. **Install MetaDrive**:
   ```bash
   pip install metadrive-simulator
   ```

2. **Install additional dependencies**:
   ```bash
   pip install pillow numpy stable-baselines3[extra] tensorboard
   ```

3. **Verify the installation**:
   ```bash
   python test_windows_metadrive.py
   ```

### For Linux/WSL2 Users (DreamerV3)

1. **Install WSL2** (if on Windows):
   ```bash
   wsl --install -d Ubuntu
   ```

2. **Install dependencies in WSL2**:
   ```bash
   pip install metadrive-simulator pillow numpy elements-jax portal-jax
   ```

3. **Verify the installation**:
   ```bash
   python test_metadrive_env.py
   ```

## Environment Configuration

The lane keeping environment is configured with:

- **Map**: Multi-lane highway with curves and traffic
- **Task**: Stay in lane while making progress toward destination
- **Observations**:
  - `image`: 64x64x3 RGB camera view
  - `speed`: Current vehicle speed (m/s)
  - `acceleration`: Current acceleration (m/s²) 
  - `angular_velocity`: [roll_rate, pitch_rate, yaw_rate] (rad/s)
  - `steering`: Current steering input [-1, 1]
  - `throttle_brake`: Current throttle/brake input [-1, 1]
  - `distance_to_route`: Lateral distance from lane center (m)
  - `route_completion`: Progress toward destination [0, 1]

- **Actions**:
  - `steering`: Steering wheel angle [-1, 1]
  - `throttle_brake`: Throttle (positive) or brake (negative) [-1, 1]

## Training

### Windows Training (Stable-Baselines3)

#### Quick Start
```bash
python train_windows_metadrive.py
```

This will:
- Create a PPO agent optimized for lane keeping
- Train with 4 parallel environments  
- Save models to `./models/`
- Log training progress to `./tensorboard_logs/`

#### Monitor Training
```bash
tensorboard --logdir=./tensorboard_logs/
```

#### Evaluate Trained Model
```bash
python eval_trained_model.py --model ./models/metadrive_ppo_final --episodes 10
```

### Linux/WSL2 Training (DreamerV3)

#### Quick Start
```bash
python train_metadrive.py
```

#### Manual Training
```bash
python dreamerv3/main.py --configs metadrive_lane_keeping
```

#### Custom Configuration
```bash
python dreamerv3/main.py \
  --configs metadrive_lane_keeping \
  --env.metadrive.length 2000 \
  --run.steps 5e6 \
  --run.envs 16
```

## Configuration Options

The environment can be customized through the config system:

```yaml
metadrive_lane_keeping:
  task: metadrive_lane_keeping
  env.metadrive:
    size: [64, 64]        # Image size
    repeat: 2             # Action repeat
    length: 1000          # Episode length
  run:
    steps: 2e6           # Training steps
    train_ratio: 64      # Training frequency
    envs: 8              # Parallel environments
    eval_envs: 2         # Evaluation environments
```

## Reward Structure

The environment provides shaped rewards for lane keeping:

- **Driving Progress**: +2.0 per meter of forward progress
- **Speed Maintenance**: +0.5 * (speed/max_speed)
- **Lane Keeping**: Lateral position penalty encourages staying in lane center
- **Success**: +50.0 for reaching destination
- **Penalties**:
  - Out of road: -20.0
  - Vehicle crash: -30.0
  - Object crash: -15.0

## Waypoints and Navigation

The environment automatically sets:
- **Start Point**: Beginning of the route
- **End Point**: Destination several kilometers away
- **Waypoints**: Intermediate navigation points along the route
- **Lane Centers**: Target trajectory for lane keeping

## Monitoring Training

Monitor training progress through:
- **TensorBoard**: Logs in `~/logdir/{timestamp}/` 
- **Console Output**: Real-time metrics
- **JSON Logs**: Detailed episode statistics

Key metrics to watch:
- `episode/score`: Cumulative reward per episode
- `episode/length`: Episode duration
- `train/loss/model`: Model learning progress
- `train/loss/policy`: Policy optimization

## Troubleshooting

### Common Issues

**ImportError: No module named 'metadrive'**
```bash
pip install metadrive-simulator
```

**Rendering issues**
- The environment is configured for headless operation by default
- For visualization, set `use_render=True` in the environment config

**Performance issues**
- Reduce `env.metadrive.repeat` for faster training
- Decrease `run.envs` if running out of memory
- Use smaller image size like [32, 32] for faster processing

### Debug Mode

Run with debug configuration:
```bash
python dreamerv3/main.py --configs metadrive_lane_keeping debug
```

This uses smaller models and shorter episodes for quick testing.

## Advanced Usage

### Custom Maps and Scenarios

Modify the environment configuration in `embodied/envs/metadrive_lane_keeping.py`:

```python
config = dict(
    map=7,  # Different map layouts (1-7)
    map_config={
        "lane_num": 3,      # Number of lanes
        "lane_width": 3.5,  # Lane width in meters
        "exit_length": 200, # Route length
    },
    traffic_density=0.2,    # More/less traffic
)
```

### Multi-Modal Observations

Add LIDAR or other sensors by modifying the sensor configuration:

```python
sensors=dict(
    rgb_camera=(RGBCamera, size[0], size[1]),
    lidar=(Lidar, num_lasers, distance),
)
```

## File Structure

```
dreamerv3-main/
├── embodied/
│   └── envs/
│       └── metadrive_lane_keeping.py  # Environment implementation
├── dreamerv3/
│   ├── main.py                        # Updated with MetaDrive registration
│   └── configs.yaml                   # Updated with MetaDrive config
├── test_metadrive_env.py              # Test script
├── train_metadrive.py                 # Training launcher
└── METADRIVE_README.md                # This file
```

## Next Steps

1. **Test the environment**: `python test_metadrive_env.py`
2. **Start training**: `python train_metadrive.py`  
3. **Monitor progress**: Check logs in `~/logdir/`
4. **Customize**: Modify configs for your specific needs

Happy training! 🚗🧠