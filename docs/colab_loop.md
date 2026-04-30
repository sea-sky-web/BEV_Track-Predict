# Colab Closed-Loop Usage

## Cell 1: 拉取代码
```python
%cd /content
!rm -rf BEV_Track-Predict
!git clone https://github.com/sea-sky-web/BEV_Track-Predict.git
%cd /content/BEV_Track-Predict
!pip -q install -r requirements.txt
```

## Cell 2: 挂载 Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 3: 解压数据集
```python
%cd /content/BEV_Track-Predict
!rm -rf wildtrack wiltrack
!unzip -q "/content/drive/MyDrive/Colab_Notebooks/dataSet/wildtrack.zip" -d .
!ls wildtrack | head
```

## Cell 4: 运行训练
```python
!python scripts/run_colab_exp.py
```

## Cell 5: 提交训练结果
```python
import os
os.environ["GITHUB_TOKEN"] = "<fill-in-token>"
os.environ["GITHUB_USER"] = "sea-sky-web"
os.environ["GITHUB_REPO"] = "BEV_Track-Predict"
os.environ["GITHUB_BRANCH"] = "main"
!python scripts/commit_ai_runs.py
```

> 不要提交含真实 token 的 notebook。
