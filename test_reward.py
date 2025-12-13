#!/usr/bin/env python3
"""
测试CARLA奖励函数的脚本

使用方法：
1. 启动CARLA服务器: ./CarlaUE4.sh
2. 启动Python37服务端: cd python37 && python carla0915.py 
3. 运行此测试脚本: python test_reward.py
"""

import socket
import json
import time
import numpy as np

def test_carla_reward():
    """测试CARLA环境的奖励函数"""
    
    # 连接到服务器
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', 50037))
        print("Connected to CARLA server")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return
    
    def send_recv(msg):
        sock.sendall(json.dumps(msg).encode())
        data = sock.recv(65536)
        return json.loads(data.decode())
    
    # 重置环境
    print("\n=== Testing RESET ===")
    reset_response = send_recv({'cmd': 'reset'})
    print(f"Reset response keys: {reset_response['obs'].keys()}")
    print(f"Initial reward: {reset_response['obs'].get('reward', 'N/A')}")
    print(f"Initial speed: {reset_response['obs'].get('speed', 'N/A')} km/h")
    
    # 测试不同的动作
    test_actions = [
        {'steer': 0.0, 'acc': 0.5, 'name': 'Forward'},
        {'steer': 0.2, 'acc': 0.5, 'name': 'Turn right while accelerating'},
        {'steer': -0.2, 'acc': 0.5, 'name': 'Turn left while accelerating'},
        {'steer': 0.0, 'acc': -0.3, 'name': 'Braking'},
        {'steer': 0.0, 'acc': 0.0, 'name': 'No action'},
        {'steer': 0.8, 'acc': 0.8, 'name': 'Sharp turn + acceleration'},
    ]
    
    print("\n=== Testing ACTIONS ===")
    for i, action in enumerate(test_actions):
        print(f"\n--- Test {i+1}: {action['name']} ---")
        response = send_recv({
            'cmd': 'step',
            'action': {'steer': action['steer'], 'acc': action['acc']}
        })
        
        obs = response['obs']
        print(f"Speed: {obs.get('speed', 0):.2f} km/h")
        print(f"Reward: {obs.get('reward', 0):.3f}")
        print(f"Done: {obs.get('done', False)}")
        
        # 如果episode结束，重新开始
        if obs.get('done', False):
            print("Episode ended, resetting...")
            send_recv({'cmd': 'reset'})
        
        time.sleep(0.1)  # 小延迟以观察效果
    
    # 连续测试以观察奖励演变
    print("\n=== Testing CONTINUOUS FORWARD MOVEMENT ===")
    send_recv({'cmd': 'reset'})  # 重置
    
    for step in range(20):
        response = send_recv({
            'cmd': 'step', 
            'action': {'steer': 0.0, 'acc': 0.3}  # 稳定前进
        })
        
        obs = response['obs']
        print(f"Step {step+1:2d}: Speed={obs.get('speed', 0):5.1f} km/h, "
              f"Reward={obs.get('reward', 0):6.3f}, Done={obs.get('done', False)}")
        
        if obs.get('done', False):
            print("Episode ended!")
            break
    
    sock.close()
    print("\nTest completed!")

if __name__ == '__main__':
    test_carla_reward()