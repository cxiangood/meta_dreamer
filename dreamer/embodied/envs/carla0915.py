import socket
import json
import numpy as np
import elements

class Carla0915Env:
    """
    DreamerV3 训练端 socket 客户端环境（与 Python 3.7 服务端通信）
    """
    def __init__(self, host, port=50037, size=(64, 64), fps=10, **kwargs):
        self.host = host
        self.port = port
        self.size = size
        self.fps = fps
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.done = True

    @property
    def obs_space(self):
        return {
            'image': elements.Space(np.uint8, self.size + (3,)),
            'speed': elements.Space(np.float32, ()),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'reset': elements.Space(bool),
            'steer': elements.Space(np.float32, (), -1.0, 1.0),
            'acc': elements.Space(np.float32, (), -1.0, 1.0),
        }

    def step(self, action):
        # 转换所有 numpy 类型为标准类型，确保可 json 序列化
        action = {k: (float(v) if isinstance(v, (np.ndarray, np.generic)) else v) for k, v in action.items()}
        if action.get('reset', False) or self.done:
            self._send({'cmd': 'reset'})
            recv = self._recv()
            obs = recv['obs'] if recv and 'obs' in recv else self._safe_obs()
            self.done = False
            # 使用服务端计算的奖励
            reward = obs.get('reward', 0.0)
            return self._format_obs(obs, reward, is_first=True)
        self._send({'cmd': 'step', 'action': {'steer': action['steer'], 'acc': action['acc']}})
        recv = self._recv()
        obs = recv['obs'] if recv and 'obs' in recv else self._safe_obs()
        self.done = obs.get('done', False)
        # 使用服务端计算的奖励
        reward = obs.get('reward', 0.0)
        return self._format_obs(obs, reward, is_last=self.done, is_terminal=self.done)
    def _safe_obs(self):
        # 返回结构完整的安全 obs，防止客户端崩溃
        print("Warning: Received incomplete observation, returning safe default.")
        return {
            'image': np.zeros(self.size + (3,), dtype=np.uint8).tolist(),
            'speed': 0.0,
            'reward': -10.0,  # 连接失败时给予惩罚
            'done': True
        }

    def _format_obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        image = np.array(obs['image'], dtype=np.uint8).reshape(self.size + (3,))
        speed = np.float32(obs.get('speed', 0.0))
        return dict(
            image=image,
            speed=speed,
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal
        )

    def _send(self, msg):
        self.sock.sendall(json.dumps(msg).encode())

    def _recv(self):
        chunks = []
        while True:
            chunk = self.sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            try:
                return json.loads(b''.join(chunks).decode())
            except json.JSONDecodeError:
                continue

    def close(self):
        self.sock.close()

# Convenience function

def make_carla0915_env(**kwargs):
    return Carla0915Env(**kwargs)
