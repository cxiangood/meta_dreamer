#!/usr/bin/env python3
"""
CARLA环境验证脚本

用于验证CARLA是否正确安装以及服务端代码是否能正常运行
"""

import sys
import subprocess
import importlib.util

def check_carla_installation():
    """检查CARLA是否正确安装"""
    print("=== 检查CARLA安装状态 ===")
    
    try:
        import carla
        print("✓ CARLA Python API已安装")
        print(f"  版本信息: {carla.__file__}")
        return True
    except ImportError as e:
        print("✗ CARLA Python API未安装")
        print(f"  错误: {e}")
        print("\n安装方法:")
        print("1. 下载CARLA: https://github.com/carla-simulator/carla/releases")
        print("2. 安装Python API: pip install carla")
        print("   或者添加到PYTHONPATH: export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla")
        return False

def check_carla_server():
    """检查CARLA服务器是否在运行"""
    print("\n=== 检查CARLA服务器状态 ===")
    
    try:
        import carla
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        version = client.get_server_version()
        print(f"✓ CARLA服务器正在运行")
        print(f"  服务器版本: {version}")
        return True
    except Exception as e:
        print("✗ CARLA服务器未运行或无法连接")
        print(f"  错误: {e}")
        print("\n启动方法:")
        print("Linux: ./CarlaUE4.sh")
        print("Windows: CarlaUE4.exe")
        print("Docker: docker run --rm -it -p 2000-2002:2000-2002 carla/carla:latest")
        return False

def test_import_server():
    """测试服务端代码的导入"""
    print("\n=== 测试服务端代码导入 ===")
    
    try:
        # 尝试导入服务端模块
        spec = importlib.util.spec_from_file_location(
            "carla0915", "/home/xiongxi/桌面/worldmodel_dreamerv3/python37/carla0915.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✓ 服务端代码导入成功")
            return True
    except Exception as e:
        print("✗ 服务端代码导入失败")
        print(f"  错误: {e}")
        return False

def main():
    """主函数"""
    print("CARLA环境验证工具")
    print("=" * 40)
    
    # 检查各个组件
    carla_installed = check_carla_installation()
    carla_running = False
    server_import = False
    
    if carla_installed:
        carla_running = check_carla_server()
        server_import = test_import_server()
    
    # 总结
    print("\n" + "=" * 40)
    print("验证结果总结:")
    print(f"CARLA安装: {'✓' if carla_installed else '✗'}")
    print(f"CARLA服务器: {'✓' if carla_running else '✗'}")
    print(f"服务端代码: {'✓' if server_import else '✗'}")
    
    if carla_installed and carla_running and server_import:
        print("\n🎉 环境验证通过！可以运行CARLA训练")
        print("启动命令: python carla0915.py")
    else:
        print("\n❌ 环境验证失败，请根据上述提示解决问题")
    
    # 关于类型检查警告的说明
    print("\n" + "=" * 40)
    print("关于类型检查警告:")
    print("- 代码中的类型检查警告是正常的")
    print("- 这些警告不影响实际运行")
    print("- 在有CARLA环境时，代码会正常工作")
    print("- 如需禁用警告，请参考 type_ignore.py")

if __name__ == '__main__':
    main()