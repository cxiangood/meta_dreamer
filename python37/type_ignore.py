# 禁用CARLA相关的类型检查警告

# 这个文件用于配置Python类型检查器，忽略CARLA相关的类型错误
# 在实际运行时，如果CARLA正确安装，这些代码都会正常工作

# 对于VS Code/PyLance，你可以在settings.json中添加：
# "python.analysis.typeCheckingMode": "basic"
# 或者在代码中使用 # type: ignore 注释

# 对于mypy，你可以创建一个mypy.ini文件：
# [mypy]
# ignore_missing_imports = True