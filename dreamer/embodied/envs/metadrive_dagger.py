"""
MetaDrive environment with DAgger (Dataset Aggregation) support.
This implementation does NOT depend on flax or separate ActorCritic agents.
Instead, it uses expert demonstrations alongside DreamerV3's own learning.

DAgger Schedule Options:
- Linear decay: expert_prob decreases linearly from initial to final over warmup steps
- Exponential decay: expert_prob = final + (initial - final) * exp(-step / decay_rate)
- Action-based: use expert when agent action differs significantly from expert

Reference: Ross et al., "A Reduction of Imitation Learning and Structured Prediction 
to No-Regret Online Learning", AISTATS 2011
"""

import numpy as np
import elements
import embodied
from embodied.envs.metadrive_lane_keeping import MetaDriveLaneKeeping

try:
    from metadrive.policy.expert_policy import ExpertPolicy
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False


class MetaDriveDAgger(embodied.Env):
    """
    MetaDrive environment with advanced DAgger support.
    
    DAgger (Dataset Aggregation) iteratively collects expert demonstrations
    on states visited by the current policy, then trains on the aggregated dataset.
    
    Features:
    1. Decaying expert probability (linear/exponential)
    2. Action-disagreement based expert intervention
    3. Episode-based and step-based scheduling
    4. Expert action always recorded for imitation loss
    """

    def __init__(
        self,
        task=None,
        size=(64, 64),
        repeat=1,
        length=1000,
        # DAgger scheduling parameters
        expert_prob_init=1.0,      # Initial expert probability (start high for IL warmup)
        expert_prob_final=0.0,     # Final expert probability (pure RL at the end)
        expert_decay_steps=500000, # Steps to decay from init to final
        expert_decay_type='linear',# 'linear', 'exponential', 'cosine'
        # Action-based expert intervention
        action_threshold=0.0,      # If > 0, use expert when |agent_action - expert_action| > threshold
        # Legacy support
        expert_prob=None,          # If set, use fixed probability (backward compatible)
        **kwargs
    ):
        if not METADRIVE_AVAILABLE:
            raise ImportError(
                "MetaDrive is not installed. Please install it with: pip install metadrive-simulator"
            )
        
        # Base environment
        self._env = MetaDriveLaneKeeping(task, size=size, repeat=repeat, length=length, **kwargs)
        
        # DAgger scheduling
        self._use_fixed_prob = expert_prob is not None
        if self._use_fixed_prob:
            # Legacy mode: fixed probability
            self._expert_prob_init = expert_prob
            self._expert_prob_final = expert_prob
            self._expert_decay_steps = 1
        else:
            self._expert_prob_init = expert_prob_init
            self._expert_prob_final = expert_prob_final
            self._expert_decay_steps = max(1, expert_decay_steps)
        
        self._expert_decay_type = expert_decay_type
        self._action_threshold = action_threshold
        
        # Counters
        self._global_step = 0
        self._episode_count = 0
        self._episode_step = 0
        
        # Expert environment (using Expert policy)
        self._expert_env = None
        self._size = size
        self._length = length
        self._kwargs = kwargs
        
        # Initialize expert if we'll ever use it
        if self._expert_prob_init > 0 or self._action_threshold > 0:
            self._init_expert_env(size, length, **kwargs)
        
        self._random = np.random.RandomState()
        self._expert_disabled = False
        self._expert_warned = False

        # Cache for last agent action (for action-based intervention)
        self._last_agent_action = None
        
        print(f"[DAgger] Initialized with decay: {expert_decay_type}, "
              f"prob: {self._expert_prob_init:.2f} -> {self._expert_prob_final:.2f} "
              f"over {self._expert_decay_steps} steps, action_threshold: {action_threshold}")

    def _init_expert_env(self, size, length, **kwargs):
        """Initialize expert environment with Expert policy."""
        cfg = dict(
            use_render=False,
            manual_control=False,
            num_agents=1,
            traffic_density=0.1,
            num_scenarios=200,
            random_agent_model=False,
            random_lane_width=False,
            random_lane_num=False,
            random_spawn_lane_index=False,
            on_continuous_line_done=False,
            out_of_route_done=True,
            map=4,
            start_seed=0,
            image_observation=True,
            sensors=dict(rgb_camera=(RGBCamera, size[0], size[1])),
            interface_panel=["rgb_camera", "dashboard"],
            agent_policy=ExpertPolicy,
        )
        self._expert_env = MetaDriveEnv(cfg)
    def _maybe_warn_expert(self, msg):
        if not self._expert_warned:
            print(f"[DAgger] Warning: {msg}")
            self._expert_warned = True

    def _ensure_expert_ready(self):
        """Try to ensure expert env has one spawned agent."""
        if self._expert_env is None or self._expert_disabled:
            return False
        try:
            agents = getattr(self._expert_env, 'agents', None)
            if agents and len(agents) > 0:
                return True
            # Retry a few times with different seeds.
            for _ in range(3):
                seed = int(self._random.randint(0, 10_000))
                try:
                    self._expert_env.reset(seed=seed)
                except TypeError:
                    self._expert_env.reset()
                agents = getattr(self._expert_env, 'agents', None)
                if agents and len(agents) > 0:
                    self._expert_warned = False
                    return True
        except Exception:
            pass
        self._maybe_warn_expert('Expert env unavailable (agents not spawned). Fallback heuristic will be used.')
        return False

    @property
    def obs_space(self):
        spaces = dict(self._env.obs_space)
        # Add expert action to observation space
        spaces['expert_action'] = elements.Space(np.float32, (2,), -1, 1)
        spaces['use_expert'] = elements.Space(bool)
        return spaces

    @property
    def act_space(self):
        return self._env.act_space

    def _get_expert_action(self, expert_obs):
        """Get action from expert policy."""
        if self._expert_env is None or self._expert_disabled:
            # Fallback: simple lane-keeping heuristic
            return np.array([0.0, 0.5], dtype=np.float32)
        
        if not self._ensure_expert_ready():
            return np.array([0.0, 0.5], dtype=np.float32)

        try:
            # Use spawned agent directly from agent dict to avoid property asserts.
            agents = self._expert_env.agents
            agent_id = next(iter(agents.keys()))
            agent = agents[agent_id]
            action = agent.policy.act(agent_id)
            steering = float(action[0])
            throttle_brake = np.clip(float(action[1]) / 4.0, -1.0, 1.0)
            return np.array([steering, throttle_brake], dtype=np.float32)
        except AssertionError as e:
            # Retry once after reset for transient init issues.
            try:
                self._expert_env.reset()
                agents = self._expert_env.agents
                agent_id = next(iter(agents.keys()))
                agent = agents[agent_id]
                action = agent.policy.act(agent_id)
                steering = float(action[0])
                throttle_brake = np.clip(float(action[1]) / 4.0, -1.0, 1.0)
                return np.array([steering, throttle_brake], dtype=np.float32)
            except Exception:
                self._maybe_warn_expert(f"Expert action failed after reset: {e}")
                return np.array([0.0, 0.5], dtype=np.float32)
        except Exception as e:
            self._maybe_warn_expert(f"Expert action failed: {e}")
            return np.array([0.0, 0.5], dtype=np.float32)

    def _compute_expert_prob(self):
        """Compute current expert probability based on decay schedule."""
        if self._use_fixed_prob:
            return self._expert_prob_init
        
        progress = min(1.0, self._global_step / self._expert_decay_steps)
        init_p = self._expert_prob_init
        final_p = self._expert_prob_final
        
        if self._expert_decay_type == 'linear':
            # Linear decay: p(t) = init + (final - init) * t
            prob = init_p + (final_p - init_p) * progress
        
        elif self._expert_decay_type == 'exponential':
            # Exponential decay: faster initial drop, slower tail
            # p(t) = final + (init - final) * exp(-5 * t)
            prob = final_p + (init_p - final_p) * np.exp(-5.0 * progress)
        
        elif self._expert_decay_type == 'cosine':
            # Cosine annealing: smooth S-curve transition
            # p(t) = final + 0.5 * (init - final) * (1 + cos(pi * t))
            prob = final_p + 0.5 * (init_p - final_p) * (1 + np.cos(np.pi * progress))
        
        elif self._expert_decay_type == 'step':
            # Step decay: high prob for first half, then drop to final
            prob = init_p if progress < 0.5 else final_p
        
        else:
            prob = init_p  # Unknown type, use initial
        
        return np.clip(prob, 0.0, 1.0)

    def _should_use_expert(self, agent_action, expert_action):
        """
        Determine whether to use expert action.
        
        Combines:
        1. Probabilistic sampling based on current decay schedule
        2. Action-disagreement threshold (if enabled)
        """
        current_prob = self._compute_expert_prob()
        
        # Probabilistic decision
        use_expert_prob = self._random.random() < current_prob
        
        # Action-disagreement based intervention
        use_expert_action = False
        if self._action_threshold > 0 and agent_action is not None:
            # Use expert if agent action differs significantly
            action_diff = np.abs(agent_action - expert_action).mean()
            use_expert_action = action_diff > self._action_threshold
        
        # Combine: use expert if either condition is met
        return use_expert_prob or use_expert_action

    def step(self, action):
        """
        Step the environment with DAgger intervention.
        
        The expert action is always recorded in observations for imitation learning,
        but actual execution depends on the current DAgger schedule.
        """
        self._global_step += 1
        self._episode_step += 1

        # Keep expert env in sync with episode reset requests.
        if isinstance(action, dict) and action.get('reset', False) and self._expert_env is not None:
            try:
                self._expert_env.reset()
            except Exception as e:
                print(f"[DAgger] Warning: Expert env reset in step failed: {e}")
        
        # Extract agent action
        if isinstance(action, dict):
            agent_action = np.array([
                action.get('steering', 0.0),
                action.get('throttle_brake', 0.0)
            ], dtype=np.float32)
        else:
            agent_action = np.array(action, dtype=np.float32)[:2]
        
        # Get expert action (always needed for recording)
        expert_action = self._get_expert_action(None)
        
        # Determine if we should use expert action
        use_expert = self._should_use_expert(agent_action, expert_action)
        
        # Choose which action to execute
        if use_expert:
            exec_action = {
                'steering': float(expert_action[0]),
                'throttle_brake': float(expert_action[1]),
                'reset': action.get('reset', False) if isinstance(action, dict) else False
            }
        else:
            exec_action = action
        
        # Step the main environment
        obs = self._env.step(exec_action)
        
        # Add DAgger information to observations
        obs['expert_action'] = expert_action.astype(np.float32)
        obs['use_expert'] = np.array(use_expert, dtype=bool)
        
        # Log progress periodically
        if self._global_step % 10000 == 0:
            current_prob = self._compute_expert_prob()
            print(f"[DAgger] Step {self._global_step}: expert_prob={current_prob:.3f}, "
                  f"episodes={self._episode_count}")
        
        return obs

    def reset(self):
        """Reset the environment."""
        return self._reset()

    def _reset(self):
        """Internal reset implementation."""
        obs = self._env._reset()
        self._episode_step = 0
        self._episode_count += 1
        
        # Reset expert env if it exists
        if self._expert_env is not None:
            try:
                self._expert_env.reset()
            except Exception as e:
                print(f"[DAgger] Warning: Expert env reset failed: {e}")
        
        # Add DAgger information to initial observation
        obs['expert_action'] = np.array([0.0, 0.0], dtype=np.float32)
        obs['use_expert'] = np.array(False, dtype=bool)
        
        return obs

    def close(self):
        """Close the environment."""
        self._env.close()
        if self._expert_env is not None:
            self._expert_env.close()

    def render(self):
        """Render the environment."""
        return self._env.render()
