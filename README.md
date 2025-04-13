# 手工神经网络实现 - CIFAR-10图像分类

本项目从零开始实现了多层感知机(MLP)分类器，用于CIFAR-10数据集的图像分类任务。不使用自动微分框架，仅使用NumPy实现神经网络的前向传播和反向传播。

## 项目结构

```
project/
├── models/
│   ├── __init__.py        # 模型包初始化
│   ├── mlp.py             # 多层感知机模型
│   └── layers.py          # 基础层实现(全连接等)
├── utils/
│   ├── __init__.py        # 工具包初始化
│   ├── activations.py     # 激活函数实现
│   ├── losses.py          # 损失函数实现
│   ├── optimizers.py      # 优化器实现(SGD、Adam)
│   ├── schedulers.py      # 学习率调度器实现
│   └── data_utils.py      # 数据处理工具
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── hyperparameter_search.py # 超参数搜索脚本
├── report.md              # 实验报告
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明
```

## 功能特点

- **模型实现**：
  - 多层感知机(MLP)
  - 可配置的隐藏层大小和激活函数

- **损失函数**：
  - 交叉熵损失
  - L2正则化

- **优化器**：
  - SGD（支持动量）
  - Adam

- **学习率调度**：
  - 线性衰减
  - 余弦衰减
  - Warmup预热

- **评估指标**：
  - 准确率(Accuracy)
  - 损失曲线和准确率曲线可视化
  - 权重参数可视化

## 环境需求

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) - Python包管理器和环境管理工具
- NumPy
- Matplotlib

### 环境设置

本项目使用UV进行环境管理。如果你还没有安装UV，可以按照[UV官方文档](https://github.com/astral-sh/uv)进行安装。

1. 创建虚拟环境并安装依赖:

```bash
# 创建虚拟环境
uv venv -p 3.10

# 激活虚拟环境
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 安装依赖
uv run hello.py
```

## 数据集

本项目使用CIFAR-10数据集，包含10个类别的60,000张32x32彩色图像，其中50,000张用于训练，10,000张用于测试。

数据集目录结构:

```
cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```

## 使用方法

### 训练模型

训练MLP模型:

```bash
uv run train.py --model mlp --hidden_dims 2048 512 --activation relu --learning_rate 0.0005 --optimizer adam --scheduler cosine --num_epochs 50
```

### 测试模型

测试MLP模型:

```bash
uv run test.py --model mlp --hidden_dims 2048 512 --activation relu --visualize
```

### 超参数搜索

对MLP进行超参数搜索:

```bash
uv run hyperparameter_search.py --model mlp --search_epochs 20 --optimizer adam
```

## 命令行参数

### 通用参数

- `--data_dir`: 数据集目录路径，默认为'cifar-10-batches-py'
- `--val_split`: 验证集比例，默认为0.1
- `--batch_size`: 批大小，默认为128
- `--seed`: 随机种子，默认为42
- `--save_dir`: 保存结果的目录，默认为'results'

### 模型参数

- `--model`: 模型类型，'mlp'
- `--hidden_dims`: 隐藏层维度，默认为[512, 128]
- `--activation`: 激活函数，'relu'、'sigmoid'或'tanh'，默认为'relu'
- `--weight_scale`: 权重初始化缩放因子，默认为1e-3

### 训练参数

- `--num_epochs`: 训练轮数，默认为50
- `--optimizer`: 优化器，'sgd'或'adam'，默认为'sgd'
- `--learning_rate`: 初始学习率，默认为0.01
- `--min_lr`: 最小学习率，默认为1e-4
- `--momentum`: SGD动量系数，默认为0.9
- `--beta1`: Adam一阶矩估计的指数衰减率，默认为0.9
- `--beta2`: Adam二阶矩估计的指数衰减率，默认为0.999
- `--scheduler`: 学习率调度器，'constant'、'linear'、'cosine'或'warmup'，默认为'constant'
- `--warmup_epochs`: 预热训练的轮数，默认为5
- `--reg_strength`: L2正则化强度，默认为1e-4
- `--patience`: 早停耐心值，0表示不使用早停，默认为10。当验证准确率连续指定轮数未提高时停止训练。

## 示例结果

使用默认参数训练的MLP模型在CIFAR-10测试集上的准确率:

- MLP: 约54%

可以通过超参数搜索获得更好的性能，典型的超参数组合:

### MLP最佳超参数
- 隐藏层: [2048, 512]
- 激活函数: ReLU
- 学习率: 0.0005
- 权重初始化缩放: 1e-3
- 正则化强度: 1e-4
- 优化器: Adam
