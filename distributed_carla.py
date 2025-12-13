#!/usr/bin/env python3
"""
分布式CARLA训练脚本 - 适用于超算多节点部署

支持:
1. 多GPU并行训练
2. 多CARLA实例
3. 分布式经验收集
4. 容错机制
"""

import os
import sys
import json
import time
import socket
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any
import argparse

class SuperComputerCARLAManager:
    """超算CARLA管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.carla_processes = []
        self.server_processes = []
        self.training_process = None
        
    def setup_environment(self):
        """设置环境变量"""
        print("设置超算环境...")
        
        # GPU设置
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
        # 虚拟显示设置
        os.environ['DISPLAY'] = ':99'
        os.environ['SDL_VIDEODRIVER'] = 'offscreen'
        os.environ['__GL_SYNC_TO_VBLANK'] = '0'
        
        # CARLA设置
        os.environ['UE4_ROOT'] = ''
        
        # 启动虚拟显示
        self.start_virtual_display()
    
    def start_virtual_display(self):
        """启动虚拟显示"""
        try:
            subprocess.Popen([
                'Xvfb', ':99', 
                '-screen', '0', '1024x768x24',
                '-ac', '+extension', 'GLX', '+render', '-noreset'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            print("✓ 虚拟显示启动成功")
        except Exception as e:
            print(f"✗ 虚拟显示启动失败: {e}")
    
    def start_carla_servers(self, num_instances: int = 1):
        """启动多个CARLA服务器实例"""
        print(f"启动 {num_instances} 个CARLA服务器实例...")
        
        carla_path = self.config.get('carla_path', './CarlaUE4.sh')
        
        for i in range(num_instances):
            port = 2000 + i * 10  # 端口间隔
            
            cmd = [
                carla_path,
                '-opengl',
                '-quality-level=Low',
                f'-world-port={port}',
                f'-rpc-port={port}',
                '-no-rendering-mode' if self.config.get('headless', True) else '',
            ]
            
            # 过滤空字符串
            cmd = [arg for arg in cmd if arg]
            
            try:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                self.carla_processes.append({
                    'process': process,
                    'port': port,
                    'pid': process.pid
                })
                print(f"  ✓ CARLA实例 {i+1} 启动 (PID: {process.pid}, Port: {port})")
                
            except Exception as e:
                print(f"  ✗ CARLA实例 {i+1} 启动失败: {e}")
        
        # 等待CARLA服务器启动
        print("等待CARLA服务器完全启动...")
        time.sleep(30)
    
    def start_python_servers(self):
        """启动Python服务端"""
        print("启动Python服务端...")
        
        python_script = self.config.get('python_server', './python37/carla0915.py')
        
        for i, carla_info in enumerate(self.carla_processes):
            server_port = 50037 + i
            
            # 修改服务端配置以连接到对应的CARLA实例
            env = os.environ.copy()
            env['CARLA_PORT'] = str(carla_info['port'])
            env['SERVER_PORT'] = str(server_port)
            
            try:
                process = subprocess.Popen(
                    [sys.executable, python_script],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                self.server_processes.append({
                    'process': process,
                    'port': server_port,
                    'carla_port': carla_info['port'],
                    'pid': process.pid
                })
                print(f"  ✓ Python服务端 {i+1} 启动 (PID: {process.pid}, Port: {server_port})")
                
            except Exception as e:
                print(f"  ✗ Python服务端 {i+1} 启动失败: {e}")
        
        time.sleep(10)
    
    def start_distributed_training(self):
        """启动分布式训练"""
        print("启动分布式DreamerV3训练...")
        
        training_script = self.config.get('training_script', './dreamerv3/dreamerv3/main.py')
        
        # 构建训练命令
        cmd = [
            sys.executable, training_script,
            '--configs', 'carla0915',
            '--task', 'carla0915_keeping',
            '--logdir', self.config.get('logdir', './logs/supercomputer'),
            '--run.envs', str(len(self.server_processes)),
            '--run.steps', str(self.config.get('training_steps', 1000000)),
        ]
        
        # 添加其他参数
        if self.config.get('debug', False):
            cmd.extend(['--configs', 'debug'])
        
        try:
            self.training_process = subprocess.Popen(cmd)
            print(f"  ✓ 训练进程启动 (PID: {self.training_process.pid})")
            
            # 等待训练完成
            self.training_process.wait()
            print("✓ 训练完成")
            
        except Exception as e:
            print(f"✗ 训练启动失败: {e}")
    
    def monitor_processes(self):
        """监控进程状态"""
        print("\n=== 进程监控 ===")
        
        # 检查CARLA进程
        alive_carla = sum(1 for p in self.carla_processes if p['process'].poll() is None)
        print(f"CARLA进程: {alive_carla}/{len(self.carla_processes)} 运行中")
        
        # 检查Python服务端
        alive_servers = sum(1 for p in self.server_processes if p['process'].poll() is None)
        print(f"Python服务端: {alive_servers}/{len(self.server_processes)} 运行中")
        
        # 检查训练进程
        if self.training_process:
            status = "运行中" if self.training_process.poll() is None else "已结束"
            print(f"训练进程: {status}")
    
    def cleanup(self):
        """清理所有进程"""
        print("\n清理进程...")
        
        # 停止训练进程
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
            print("  ✓ 训练进程已停止")
        
        # 停止Python服务端
        for server in self.server_processes:
            if server['process'].poll() is None:
                server['process'].terminate()
        print(f"  ✓ {len(self.server_processes)} 个Python服务端已停止")
        
        # 停止CARLA进程
        for carla in self.carla_processes:
            if carla['process'].poll() is None:
                carla['process'].terminate()
        print(f"  ✓ {len(self.carla_processes)} 个CARLA实例已停止")
    
    def run(self):
        """运行完整的训练流程"""
        try:
            self.setup_environment()
            self.start_carla_servers(self.config.get('num_carla_instances', 1))
            self.start_python_servers()
            
            # 监控一次状态
            self.monitor_processes()
            
            # 开始训练
            self.start_distributed_training()
            
        except KeyboardInterrupt:
            print("\n收到中断信号...")
        except Exception as e:
            print(f"\n运行出错: {e}")
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description='超算CARLA分布式训练')
    parser.add_argument('--config', type=str, default='supercomputer_config.json',
                       help='配置文件路径')
    parser.add_argument('--num-instances', type=int, default=1,
                       help='CARLA实例数量')
    parser.add_argument('--training-steps', type=int, default=1000000,
                       help='训练步数')
    parser.add_argument('--logdir', type=str, default='./logs/supercomputer',
                       help='日志目录')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {
        'carla_path': os.path.expanduser('~/CARLA_0.9.15/CarlaUE4.sh'),
        'python_server': './python37/carla0915.py',
        'training_script': './dreamerv3/dreamerv3/main.py',
        'num_carla_instances': args.num_instances,
        'training_steps': args.training_steps,
        'logdir': args.logdir,
        'debug': args.debug,
        'headless': True,
    }
    
    # 如果配置文件存在，加载它
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    print("超算CARLA分布式训练系统")
    print("=" * 40)
    print(f"CARLA实例数: {config['num_carla_instances']}")
    print(f"训练步数: {config['training_steps']}")
    print(f"日志目录: {config['logdir']}")
    print(f"调试模式: {config['debug']}")
    print("=" * 40)
    
    # 创建管理器并运行
    manager = SuperComputerCARLAManager(config)
    manager.run()

if __name__ == '__main__':
    main()