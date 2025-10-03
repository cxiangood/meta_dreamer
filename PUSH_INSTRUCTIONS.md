# Git Push Instructions - 推送到GitHub的说明

由于网络连接问题，自动推送失败。请按照以下步骤手动推送代码到GitHub：

## 方法1：配置代理后推送（推荐）

如果你有HTTP/HTTPS代理：

```bash
cd d:\学习\智能驾驶训练集生成\Prj_worldmoudle

# 配置代理（替换为你的代理地址）
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 推送
git push -u origin main

# 推送后取消代理配置（可选）
git config --global --unset http.proxy
git config --global --unset https.proxy
```

## 方法2：使用GitHub Desktop（最简单）

1. 下载安装 [GitHub Desktop](https://desktop.github.com/)
2. 在GitHub Desktop中选择 "Add Local Repository"
3. 选择文件夹：`d:\学习\智能驾驶训练集生成\Prj_worldmoudle`
4. 点击 "Publish repository"
5. 填写：
   - Name: `meta_dreamer`
   - Description: DreamerV3 + MetaDrive integration project
   - ✅ Keep this code private (如果需要私有仓库)
6. 点击 "Publish Repository"

## 方法3：配置SSH密钥

如果SSH端口22被封：

```bash
# 编辑 ~/.ssh/config 文件
Host github.com
    Hostname ssh.github.com
    Port 443
    User git

# 然后推送
cd d:\学习\智能驾驶训练集生成\Prj_worldmoudle
git push -u origin main
```

## 方法4：手动上传到GitHub Web界面

如果以上方法都不行，可以手动上传：

1. 访问 https://github.com/cxiangood/meta_dreamer
2. 点击 "Upload files"
3. 将以下文件夹拖拽上传：
   - `dreamerv3-main/`
   - `metadrive-main/`
   - `README.md`
   - `.gitignore`
   - `test_navigation.py`
   - `get-pip.py`

注意：GitHub Web界面单次最多上传100个文件，可能需要多次上传。

## 当前Git状态

```bash
Repository: d:\学习\智能驾驶训练集生成\Prj_worldmoudle
Branch: main
Remote: origin -> git@github.com:cxiangood/meta_dreamer.git
Commit: ✅ "Initial commit: DreamerV3 + MetaDrive integration"
Files: 1552 files committed
Status: Ready to push
```

## 验证推送成功

推送成功后，访问以下链接验证：

https://github.com/cxiangood/meta_dreamer

你应该能看到：
- ✅ README.md 显示项目介绍
- ✅ dreamerv3-main/ 文件夹
- ✅ metadrive-main/ 文件夹
- ✅ 1552 files committed

## 后续提交

推送成功后，后续修改代码可以这样提交：

```bash
cd d:\学习\智能驾驶训练集生成\Prj_worldmoudle

# 查看修改
git status

# 添加修改的文件
git add .

# 提交
git commit -m "描述你的修改"

# 推送
git push
```

## 故障排除

### 问题1: "Failed to connect to github.com port 443"
**解决**: 使用方法1配置代理，或使用方法2的GitHub Desktop

### 问题2: "ssh: connect to host github.com port 22: Connection timed out"
**解决**: 使用方法3配置SSH端口443

### 问题3: "authentication failed"
**解决**: 
- HTTPS方式需要Personal Access Token (不是密码)
- SSH方式需要配置SSH密钥: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

## 需要帮助？

如果遇到问题，可以：
1. 检查网络连接
2. 尝试不同的推送方法
3. 查看GitHub文档: https://docs.github.com/
4. 在项目Issue中提问
