# Colab自动化训练启动指南

## 前置条件

1. **已安装依赖**（已完成）
2. **Google Chrome浏览器**（必须）
3. **Google账户**（用于登录Colab）
4. **GitHub仓库推送权限**（用于自动推送代码）

---

## 启动步骤

### 第一步：启动带Remote Debugging的Chrome

**Windows用户：**

1. **关闭所有Chrome窗口**（必须完全关闭）
2. 打开命令提示符（CMD）或PowerShell
3. 执行以下命令启动Chrome：

```cmd
# 方法1：如果Chrome在默认位置
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

# 方法2：如果Chrome不在默认位置，找到chrome.exe路径
# 例如：
# "C:\Users\YourName\AppData\Local\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
```

**Mac/Linux用户：**

```bash
google-chrome --remote-debugging-port=9222
```

### 第二步：在Chrome中登录Google账户

1. 在刚启动的Chrome浏览器中打开 [Google Colab](https://colab.research.google.com)
2. 使用你的Google账户登录
3. **保持这个Chrome窗口打开**（不要关闭）

### 第三步：运行自动化脚本

1. 确保已激活虚拟环境：
```bash
cd c:\Users\zhangweichao\Desktop\BEV_Track&Predict
.\venv\Scripts\Activate.ps1
```

2. 运行测试脚本验证连接：
```bash
python test_browser_control.py
```

3. 运行完整自动化训练：
```bash
python run_automated_training.py
```

---

## 完整启动流程示例

```bash
# 1. 打开新的命令提示符窗口，启动带调试端口的Chrome
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

# 2. 在Chrome中登录Google账户并打开Colab

# 3. 打开另一个命令提示符窗口
cd c:\Users\zhangweichao\Desktop\BEV_Track&Predict
.\venv\Scripts\Activate.ps1
python run_automated_training.py
```

---

## 配置说明

配置文件位于 `colab_automation/.env`：

```ini
# CDP连接配置
PLAYWRIGHT_ATTACH_EXISTING_CHROME=true  # 使用已登录的Chrome
PLAYWRIGHT_CDP_URL=http://localhost:9222  # CDP端口

# 训练参数
TRAIN_EPOCHS=10
TRAIN_BATCH=1
TRAIN_MAX_FRAMES=300
```

---

## 常见问题

### Q1: 连接到Chrome失败

**错误信息：** `无法连接到Chrome浏览器`

**解决方案：**
1. 确保所有Chrome窗口已关闭
2. 使用正确的命令重新启动Chrome
3. 检查端口9222是否被其他程序占用

### Q2: Colab提示需要登录

**错误信息：** 页面跳转到Google登录页面

**解决方案：**
1. 确保在带调试端口的Chrome中已登录Google账户
2. 先手动打开Colab确认登录状态正常
3. 重新运行自动化脚本

### Q3: 训练单元执行超时

**解决方案：**
1. 增加超时时间：修改 `.env` 中的 `PLAYWRIGHT_TIMEOUT=600000`（10分钟）
2. 确保网络连接稳定

### Q4: Git推送失败

**解决方案：**
1. 确保Git已配置用户信息：
```bash
git config --global user.email "your_email@example.com"
git config --global user.name "Your Name"
```
2. 确保已配置SSH密钥或HTTPS凭据

---

## 日志文件

- `automated_training.log` - 自动化训练主日志
- `colab_automation.log` - 自动化模块日志
- `errors/report_*.txt` - 执行报告
- `errors/errors_*.json` - 错误详情

---

## 注意事项

1. **不要关闭手动启动的Chrome窗口**，否则自动化会失败
2. **保持网络连接稳定**，训练过程中网络中断会导致失败
3. **首次运行建议使用测试脚本**验证配置正确
4. 如果遇到登录问题，重新启动Chrome并重新登录
