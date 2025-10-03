#!/usr/bin/env python3
"""测试MetaDrive导航模块是否正确配置"""
import sys
import os
sys.path.insert(0, 'dreamerv3-main/dreamerv3-main')

# 启用渲染
os.environ['METADRIVE_RENDER'] = '1'

from embodied.envs.metadrive_lane_keeping import MetaDriveLaneKeeping

print("=" * 60)
print("测试MetaDrive导航标记显示")
print("=" * 60)

# 创建环境
print("\n[1/3] 创建环境...")
env = MetaDriveLaneKeeping("lane_keeping", size=(64, 64))
print("✓ 环境创建成功")

# 重置环境
print("\n[2/3] 重置环境...")
obs = env.step({"reset": True, "steering": 0.0, "throttle_brake": 0.0})
print("✓ 环境重置成功")

# 检查导航模块
print("\n[3/3] 检查导航模块...")
agent = env._env.agent
print(f"  - Agent: {agent}")
print(f"  - Navigation module: {agent.navigation}")
print(f"  - Navigation type: {type(agent.navigation).__name__}")
print(f"  - Show navi mark: {agent.config.get('show_navi_mark', 'Not set')}")
print(f"  - Show dest mark: {agent.config.get('show_dest_mark', 'Not set')}")
print(f"  - Navigation module config: {agent.config.get('navigation_module', 'Not set')}")

if agent.navigation is not None:
    print("\n✅ 导航模块已正确配置！")
    print("💡 提示：如果渲染窗口已打开，你应该能看到：")
    print("   - 🔵 蓝色方块：waypoint标记")
    print("   - 🔴 红色标记：目的地")
    print("   - 📏 绿色/蓝色线：导航连线")
else:
    print("\n❌ 导航模块未配置")
    print("请检查配置")

# 运行几步看看
print("\n[测试] 运行10步...")
for i in range(10):
    obs = env.step({
        "reset": False,
        "steering": 0.0,
        "throttle_brake": 0.3  # 轻微加速
    })
    if i == 0:
        print(f"  Step {i+1}: Speed={obs.get('speed', 0):.2f} m/s")
    elif i == 9:
        print(f"  Step {i+1}: Speed={obs.get('speed', 0):.2f} m/s")

print("\n✓ 测试完成！请查看渲染窗口以确认导航标记是否显示")
print("=" * 60)
