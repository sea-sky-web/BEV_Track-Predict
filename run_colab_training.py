"""
在Google Colab环境中运行轻量化多视角检测（MVDet）模型训练的脚本
"""

import subprocess
import sys
from pathlib import Path

def run_colab_training():
    """在Colab环境中运行训练"""
    try:
        # 检查是否在Colab环境中
        import google.colab
        IN_COLAB = True
    except ImportError:
        print("错误: 请在Google Colab环境中运行此脚本")
        return False
    
    try:
        # 克隆仓库
        print("克隆仓库...")
        subprocess.run(["git", "clone", "https://github.com/sea-sky-web/BEV_Track-Predict.git"], check=True)
        
        # 进入项目目录
        print("进入项目目录...")
        import os
        os.chdir("BEV_Track-Predict")
        
        # 安装依赖
        print("安装依赖库...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        # 检查CUDA可用性
        import torch
        if not torch.cuda.is_available():
            print("警告: CUDA不可用，将使用CPU进行训练（速度会很慢）")
        else:
            print(f"CUDA设备信息: {torch.cuda.get_device_name(0)}")
        
        # 运行训练命令
        print("开始训练...")
        training_command = [
            sys.executable, "src/train_main.py",
            "--data_root", "wildtrack",
            "--views", "0,1,2",
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--epochs", "50",
            "--batch", "2",
            "--amp"
        ]
        
        print(f"执行命令: {' '.join(training_command)}")
        result = subprocess.run(training_command, capture_output=True, text=True)
        
        # 打印训练输出
        print("\n训练输出:")
        print(result.stdout)
        
        if result.returncode != 0:
            print("\n训练过程中出现错误:")
            print(result.stderr)
            return False
        
        # 保存训练日志
        log_path = Path("training_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("训练命令: " + " ".join(training_command) + "\n\n")
            f.write("标准输出:\n" + result.stdout + "\n\n")
            f.write("标准错误:\n" + result.stderr + "\n")
        
        print(f"\n训练日志已保存到: {log_path}")
        
        # 评估模型性能
        print("\n开始模型评估...")
        evaluation_command = [
            sys.executable, "src/evaluate_main.py",
            "--data_root", "wildtrack",
            "--views", "0,1,2",
            "--device", "cuda" if torch.cuda.is_available() else "cpu"
        ]
        
        try:
            eval_result = subprocess.run(evaluation_command, capture_output=True, text=True)
            print("\n评估输出:")
            print(eval_result.stdout)
            
            if eval_result.returncode != 0:
                print("\n评估过程中出现错误:")
                print(eval_result.stderr)
            else:
                print("\n评估成功完成")
        
        except FileNotFoundError:
            print("警告: 评估脚本未找到，跳过评估")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"执行命令时出错: {e}")
        return False
    except Exception as e:
        print(f"发生意外错误: {e}")
        import traceback
        print(f"堆栈跟踪: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    if run_colab_training():
        print("\n训练过程完成！")
        print("\n接下来的步骤:")
        print("1. 查看训练日志: training_log.txt")
        print("2. 检查训练结果: outputs/ 目录")
        print("3. 评估模型性能（如果已运行评估）")
    else:
        print("\n训练过程失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
