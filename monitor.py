#!/usr/bin/env python3
"""
超算性能监控和日志分析工具

功能:
1. 实时监控GPU使用率
2. 分析训练日志和指标
3. 生成性能报告
4. 自动故障检测
"""

import os
import sys
import json
import time
import psutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class SuperComputerMonitor:
    """超算性能监控器"""
    
    def __init__(self, logdir: str):
        self.logdir = Path(logdir)
        self.start_time = datetime.now()
        
    def get_gpu_info(self) -> List[Dict]:
        """获取GPU信息"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            gpus = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            gpus.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'temperature': int(parts[2]) if parts[2] != '[N/A]' else 0,
                                'utilization': int(parts[3]) if parts[3] != '[N/A]' else 0,
                                'memory_used': int(parts[4]) if parts[4] != '[N/A]' else 0,
                                'memory_total': int(parts[5]) if parts[5] != '[N/A]' else 0,
                                'power': float(parts[6]) if parts[6] != '[N/A]' else 0.0
                            })
            return gpus
        except Exception as e:
            print(f"获取GPU信息失败: {e}")
            return []
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_count': psutil.cpu_count(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    
    def get_carla_processes(self) -> List[Dict]:
        """获取CARLA进程信息"""
        carla_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                if 'carla' in proc.info['name'].lower() or 'CarlaUE4' in proc.info['name']:
                    carla_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'running_time': time.time() - proc.info['create_time']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return carla_processes
    
    def parse_dreamerv3_logs(self) -> Dict:
        """解析DreamerV3训练日志"""
        log_data = {
            'steps': [],
            'rewards': [],
            'losses': [],
            'episodes': [],
            'fps': []
        }
        
        # 查找最新的日志文件
        log_files = list(self.logdir.glob("**/metrics.jsonl"))
        
        if not log_files:
            return log_data
        
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_log, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        
                        if 'step' in data:
                            log_data['steps'].append(data['step'])
                        
                        if 'episode_reward' in data:
                            log_data['rewards'].append(data['episode_reward'])
                        
                        if 'policy_loss' in data:
                            log_data['losses'].append(data['policy_loss'])
                        
                        if 'episode_length' in data:
                            log_data['episodes'].append(data['episode_length'])
                        
                        if 'fps' in data:
                            log_data['fps'].append(data['fps'])
        
        except Exception as e:
            print(f"解析日志失败: {e}")
        
        return log_data
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        gpus = self.get_gpu_info()
        system = self.get_system_info()
        carla_procs = self.get_carla_processes()
        training_data = self.parse_dreamerv3_logs()
        
        runtime = datetime.now() - self.start_time
        
        report = f"""
=== 超算CARLA训练性能报告 ===
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
运行时长: {runtime}

=== 系统资源 ===
CPU使用率: {system['cpu_percent']:.1f}% ({system['cpu_count']} 核心)
内存使用率: {system['memory_percent']:.1f}% ({system['memory_used_gb']:.1f}GB / {system['memory_total_gb']:.1f}GB)
磁盘使用率: {system['disk_percent']:.1f}%
系统负载: {system['load_average'][0]:.2f} {system['load_average'][1]:.2f} {system['load_average'][2]:.2f}

=== GPU状态 ==="""

        for gpu in gpus:
            memory_percent = (gpu['memory_used'] / gpu['memory_total'] * 100) if gpu['memory_total'] > 0 else 0
            report += f"""
GPU {gpu['index']} ({gpu['name']}):
  利用率: {gpu['utilization']}%
  内存: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({memory_percent:.1f}%)
  温度: {gpu['temperature']}°C
  功耗: {gpu['power']:.1f}W"""

        report += f"""

=== CARLA进程 ===
运行中的CARLA实例: {len(carla_procs)}"""

        for proc in carla_procs:
            hours = proc['running_time'] / 3600
            report += f"""
PID {proc['pid']} ({proc['name']}):
  CPU: {proc['cpu_percent']:.1f}%
  内存: {proc['memory_percent']:.1f}%
  运行时间: {hours:.1f} 小时"""

        if training_data['rewards']:
            recent_rewards = training_data['rewards'][-100:] if len(training_data['rewards']) > 100 else training_data['rewards']
            avg_reward = np.mean(recent_rewards)
            max_reward = np.max(training_data['rewards'])
            
            report += f"""

=== 训练进度 ===
总训练步数: {training_data['steps'][-1] if training_data['steps'] else 0}
最近100轮平均奖励: {avg_reward:.3f}
历史最高奖励: {max_reward:.3f}
训练集数: {len(training_data['rewards'])}"""

            if training_data['fps']:
                avg_fps = np.mean(training_data['fps'][-50:]) if len(training_data['fps']) > 50 else np.mean(training_data['fps'])
                report += f"""
平均FPS: {avg_fps:.1f}"""

        return report
    
    def plot_training_curves(self, save_path: str | None = None):
        """绘制训练曲线"""
        training_data = self.parse_dreamerv3_logs()
        
        if not training_data['rewards']:
            print("没有训练数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CARLA DreamerV3 训练监控', fontsize=16)
        
        # 奖励曲线
        if training_data['rewards']:
            axes[0, 0].plot(training_data['rewards'])
            axes[0, 0].set_title('Episode Reward')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # 损失曲线
        if training_data['losses']:
            axes[0, 1].plot(training_data['losses'])
            axes[0, 1].set_title('Policy Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Episode长度
        if training_data['episodes']:
            axes[1, 0].plot(training_data['episodes'])
            axes[1, 0].set_title('Episode Length')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].grid(True)
        
        # FPS
        if training_data['fps']:
            axes[1, 1].plot(training_data['fps'])
            axes[1, 1].set_title('Training FPS')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('FPS')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        else:
            plt.show()
    
    def check_alerts(self) -> List[str]:
        """检查告警条件"""
        alerts = []
        
        # 检查GPU温度
        gpus = self.get_gpu_info()
        for gpu in gpus:
            if gpu['temperature'] > 85:
                alerts.append(f"⚠ GPU {gpu['index']} 温度过高: {gpu['temperature']}°C")
            
            if gpu['utilization'] < 10:
                alerts.append(f"⚠ GPU {gpu['index']} 利用率过低: {gpu['utilization']}%")
        
        # 检查内存使用
        system = self.get_system_info()
        if system['memory_percent'] > 90:
            alerts.append(f"⚠ 内存使用率过高: {system['memory_percent']:.1f}%")
        
        # 检查CARLA进程
        carla_procs = self.get_carla_processes()
        if len(carla_procs) == 0:
            alerts.append("⚠ 没有检测到CARLA进程")
        
        return alerts
    
    def monitor_realtime(self, interval: int = 30):
        """实时监控"""
        print("开始实时监控 (Ctrl+C 停止)")
        print("=" * 60)
        
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # 显示报告
                print(self.generate_performance_report())
                
                # 检查告警
                alerts = self.check_alerts()
                if alerts:
                    print("\n=== 告警信息 ===")
                    for alert in alerts:
                        print(alert)
                
                print(f"\n下次更新: {interval}秒后...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")

def main():
    parser = argparse.ArgumentParser(description='超算CARLA训练监控工具')
    parser.add_argument('--logdir', type=str, default='./logs',
                       help='日志目录')
    parser.add_argument('--mode', type=str, choices=['monitor', 'report', 'plot'],
                       default='monitor', help='运行模式')
    parser.add_argument('--interval', type=int, default=30,
                       help='监控间隔(秒)')
    parser.add_argument('--output', type=str,
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    monitor = SuperComputerMonitor(args.logdir)
    
    if args.mode == 'monitor':
        monitor.monitor_realtime(args.interval)
    elif args.mode == 'report':
        report = monitor.generate_performance_report()
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"报告已保存到: {args.output}")
        else:
            print(report)
    elif args.mode == 'plot':
        monitor.plot_training_curves(args.output)

if __name__ == '__main__':
    main()