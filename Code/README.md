# MechDDI

本项目是 MechDDI 的 PyTorch 实现，包含多模态特征融合、动态机制路由和 DDI 预测模块，支持二分类、多分类和多标签任务。

## 环境

- Python 3.9+
- PyTorch 2.0+
- NumPy 1.24+

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速运行

项目自带小规模模拟数据，可用于检查代码能否正常运行：

```bash
python -m mechddi.train --epochs 5
```

## 使用 OpenDDI 数据

将 OpenDDI 数据解压后，指定数据目录和任务类型：

```bash
python -m mechddi.train \
  --openddi-root /path/to/MultiMData/datasets \
  --task multiclass \
  --max-pairs 4096 \
  --epochs 5
```

`--task` 可设置为：

- `binary`：二分类任务
- `multiclass`：多分类任务
- `multilabel`：多标签任务

`--max-pairs` 用于限制读取的药物对数量，调试时可以保留；使用完整数据时可删除该参数。

## 测试

```bash
python -m pytest -q
```

## 主要文件

```text
mechddi/model.py      模型结构
mechddi/data.py       通用数据处理
mechddi/openddi.py    OpenDDI 数据接口
mechddi/train.py      训练入口
tests/                运行测试
```
