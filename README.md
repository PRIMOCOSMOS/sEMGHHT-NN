# sEMG-HHT CNN Classifier | sEMG-HHT CNN 分类器

[English](#english) | [中文](#chinese)

---

## <a name="english"></a>English Version

A dual classification system for surface electromyography (sEMG) signals using Hilbert-Huang Transform (HHT) representation. This project separates gender and movement quality classification into specialized models for better accuracy.

### 🎯 Overview

This project implements a dual deep learning pipeline:
1. **Deep Learning CNN** for Action Quality (Full, Half, Invalid) - 3 classes
2. **SVM Classifier** for Gender Classification (M, F) - 2 classes

**Key Features:**
- ✅ **Expanded CNN architecture** (7 layers, 2048 channels) - **NEW!**
- ✅ **BatchNormalization** + Kaiming initialization for training stability
- ✅ **Learning rate warmup** + cosine annealing scheduling
- ✅ **Gradient clipping** to prevent explosion
- ✅ **Label smoothing** for better generalization
- ✅ **Separate optimized models** for each task

**📖 LATEST: [Refactoring Summary](REFACTORING_SUMMARY.md)** - **NEW!** Complete details on the expanded architecture and training optimizations.

**📖 [Dual Classifier System Guide](DUAL_CLASSIFIER_GUIDE.md)** - Complete documentation on the dual classifier system.

### 🏗️ Model Architecture

**Configurable CNN Encoder Structure (1-8 Layers):** - **CONFIGURABLE!**
- **Flexible depth** from 1 to 8 convolutional layers (default: 5)
- Each convolutional layer contains:
  - Conv2D (kernel=3, stride=2, padding=1, bias=False)
  - **Batch Normalization** (training stability)
  - LeakyReLU activation (slope=0.2)
  - **Kaiming initialization** (proper gradient flow)
- **Residual connections** in deeper layers (6+)
- **Global Average Pooling** at the end
- **Channel progression**: 64 → 128 → 256 → 512 → 1024 → 2048 → ...
- Output: **Feature vector** (size depends on num_layers)
  - 3 layers: 256-dim
  - 5 layers: 1024-dim
  - 7 layers: 2048-dim
  - 8 layers: 2048-dim

**Adaptive Classifier Head:**
- Automatically scales intermediate layer sizes based on encoder depth
- Ensures smooth gradient flow regardless of network configuration

**Classifier Options:**
1. **Action Quality CNN**: Configurable CNN → Dropout → Adaptive FC layers → 3 classes
2. **Gender SVM**: CNN features → StandardScaler → RBF SVM → 2 classes

### 📊 Classification Task

**Dual Classification System:**
- **Action Quality**: 3 classes (Full, Half, Invalid) - Deep Learning CNN
- **Gender**: 2 classes (Male, Female) - SVM

**Why Dual Classifiers?**
- Better accuracy through task-specific optimization
- Faster convergence for each simpler task
- More stable training dynamics
- Easier to debug and improve

**Class Mapping:**
| Task | Classes | Model Type |
|------|---------|------------|
| Action Quality | Full, Half, Invalid | Deep CNN (configurable 1-8 layers) **[CONFIGURABLE]** |
| Gender | M, F | SVM (RBF kernel) |

### 🆕 Recent Improvements

**2025-12-22 (Latest Update):**
1. **Configurable Network Depth** - Number of layers now configurable as hyperparameter (1-8 layers)
2. **Adaptive Architecture** - Classifier head automatically scales with encoder depth
3. **Parameter Validation** - Input validation ensures proper dimensions throughout training
4. **Safe Checkpoint Saving** - Atomic writes with disk space checks prevent corruption
5. **Command-line Control** - New `--num_encoder_layers` and `--base_channels` arguments

**Previous Improvements (2025-12-22):**

**Problem Solved:** Previous notebook had issues with loss barely decreasing and accuracy not improving.

**Key Solutions:**
1. **Expanded Network** - From 3-5 layers to **configurable depth** (default 7 layers)
2. **Better Initialization** - Kaiming initialization prevents vanishing/exploding gradients
3. **Batch Normalization** - Replaced InstanceNorm for faster, more stable training
4. **Learning Rate Strategy** - Lowered LR (0.0001) + warmup (5 epochs) + cosine annealing
5. **Gradient Clipping** - Prevents gradient explosion in deep network
6. **Label Smoothing** - Improves generalization and prevents overconfidence
7. **Residual Connections** - Better gradient flow in deeper layers
8. **AdamW Optimizer** - With weight decay for better regularization

**See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for complete details.**

### 🚀 Quick Start

#### Option 1: Kaggle Notebook (Easiest)

1. Upload `semg_hht_cnn_classifier.ipynb` to Kaggle
2. Add the **HILBERTMATRIX_NPZ** dataset to your notebook
3. Enable GPU accelerator (Settings → Accelerator → GPU)
4. Run all cells

The notebook automatically detects Kaggle environment and loads data from `/kaggle/input/hilbertmatrix-npz/hht_matrices/`

#### Option 2: Command-Line Training (For Local/Server)

```bash
# Install dependencies
pip install -r requirements.txt

# Train with your data (new dual classifier system)
python train.py --data_dir ./data --checkpoint_dir ./checkpoints --epochs 100

# Advanced training with custom parameters and configurable network depth
python train.py \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --num_encoder_layers 5 \
    --base_channels 64 \
    --test_size 0.2

# Train with deeper network (7 layers, more features)
python train.py \
    --data_dir ./data \
    --num_encoder_layers 7 \
    --base_channels 64 \
    --epochs 100

# Train with shallower network (3 layers, faster training)
python train.py \
    --data_dir ./data \
    --num_encoder_layers 3 \
    --base_channels 64 \
    --epochs 100

# Resume from checkpoint
python train.py --data_dir ./data --checkpoint_dir ./checkpoints --resume

# Run inference
python inference.py --checkpoint ./checkpoints/final --input ./new_data/
```

**New Parameters:**
- `--num_encoder_layers`: Number of CNN layers (1-8, default: 5)
  - More layers = deeper network, better feature extraction, slower training
  - Fewer layers = faster training, less memory usage
- `--base_channels`: Base number of channels (default: 64)
  - Higher values = more parameters, better capacity

See [DUAL_CLASSIFIER_GUIDE.md](DUAL_CLASSIFIER_GUIDE.md) for detailed instructions on the new architecture.

### 📁 Data Format

**File Naming Convention:**
```
MUSCLENAME_movement_GENDER_###.npz
```

Examples:
- `BICEPS_fatiguetest_M_006.npz` → Male, Full movement
- `TRICEPS_half_F_012.npz` → Female, Half movement
- `FOREARM_invalid_M_003.npz` → Male, Invalid movement
- `Test1_1_015.npz` → Unlabeled test file (starts with "Test")

**File Content:**
Each `.npz` file contains a 256×256 HHT matrix stored with key `'hht'`.

### 🔧 Module Functions

**1. `train.py` - Production Training Script**
- Loads .npz files from directory
- Parses filenames to extract gender and movement labels
- Trains CNN encoder to extract features
- Trains SVM classifier on extracted features
- Saves checkpoints (encoder, scaler, SVM, metadata)
- Evaluates on validation set
- Runs inference on test files (files starting with "Test")

**2. `inference.py` - Inference Script**
- Loads trained model from checkpoint
- Processes single file or batch of files
- Outputs predictions with confidence scores
- Saves results to JSON

**3. `generate_sample_data.py` - Data Generator**
- Creates synthetic 256×256 HHT matrices for testing
- Generates proper filename formats
- Useful for testing the pipeline

**4. Jupyter Notebook `semg_hht_cnn_classifier.ipynb`**
- Interactive exploration and visualization
- Integrated with Kaggle datasets
- Step-by-step training workflow
- Suitable for experimentation

### 📈 Training Process

1. **Data Loading**: Loads all .npz files, filters test files (starting with "Test")
2. **Feature Extraction**: CNN encoder processes 256×256 matrices → 256-dim vectors
3. **Normalization**: StandardScaler normalizes features
4. **SVM Training**: RBF kernel SVM trains on normalized features
5. **Validation**: Computes accuracy, precision, recall, F1-score
6. **Test Inference**: Predicts labels for test files
7. **Checkpoint Saving**: Saves complete model state

---

## <a name="chinese"></a>中文版本

基于卷积神经网络（CNN）的表面肌电信号（sEMG）双分类器系统，使用希尔伯特-黄变换（HHT）表示。该项目将性别和动作质量分类分离为专门的模型以获得更好的准确性。

### 🎯 概述

该项目实现了双重深度学习流程：
1. **深度学习CNN** 用于动作质量（全程、半程、无效）- 3类
2. **SVM分类器** 用于性别分类（男、女）- 2类

**主要特点：**
- ✅ **可配置的CNN架构**（1-8层，灵活深度）- **新功能！**
- ✅ **自适应分类头**根据网络深度自动调整
- ✅ **批归一化** + Kaiming初始化以提高训练稳定性
- ✅ **安全的检查点保存**防止磁盘空间不足导致的损坏
- ✅ **学习率预热** + 余弦退火调度
- ✅ **梯度裁剪**防止梯度爆炸
- ✅ **标签平滑**提高泛化能力
- ✅ 每个任务的**单独优化模型**

**📖 最新：[重构总结](REFACTORING_SUMMARY.md)** - **新！** 扩展架构和训练优化的完整细节。

**📖 [双分类器系统指南](DUAL_CLASSIFIER_GUIDE.md)** - 关于双分类器系统的完整文档。

### 🏗️ 模型架构

**可配置的CNN编码器结构（1-8层）：** - **可配置！**
- **灵活深度** 从1到8个卷积层（默认：5层）
- 每个卷积层包含：
  - Conv2D（kernel=3, stride=2, padding=1, bias=False）
  - **批归一化**（训练稳定性）
  - LeakyReLU 激活函数（slope=0.2）
  - **Kaiming初始化**（正确的梯度流动）
- **残差连接**在更深层中（第6层及以上）
- 末尾使用**全局平均池化**
- **通道递增序列**：64 → 128 → 256 → 512 → 1024 → 2048 → ...
- 输出：**特征向量**（大小取决于层数）
  - 3层：256维
  - 5层：1024维
  - 7层：2048维
  - 8层：2048维

**自适应分类头：**
- 根据编码器深度自动调整中间层大小
- 确保无论网络配置如何都能平滑的梯度流动

**分类器选项：**
1. **动作质量CNN**：可配置CNN → Dropout → 自适应全连接层 → 3类
2. **性别SVM**：CNN特征 → StandardScaler → RBF SVM → 2类

### 📊 分类任务

**双分类器系统：**
- **动作质量**：3类（全程、半程、无效）- 深度学习CNN
- **性别**：2类（男性、女性）- SVM

**为什么使用双分类器？**
- 通过特定任务优化获得更好的准确性
- 每个简单任务更快收敛
- 更稳定的训练动态
- 更容易调试和改进

**类别映射：**
| 任务 | 类别 | 模型类型 |
|------|------|----------|
| 动作质量 | 全程、半程、无效 | 深度CNN（可配置1-8层）**[可配置]** |
| 性别 | 男、女 | SVM（RBF核）|

### 🆕 最近改进

**2025-12-22（最新更新）：**
1. **可配置的网络深度** - 层数现在可作为超参数配置（1-8层）
2. **自适应架构** - 分类头根据编码器深度自动调整
3. **参数验证** - 输入验证确保训练过程中的正确维度
4. **安全的检查点保存** - 原子写入和磁盘空间检查防止损坏
5. **命令行控制** - 新增 `--num_encoder_layers` 和 `--base_channels` 参数

**之前的改进（2025-12-22）：**

**解决的问题：** 之前的笔记本存在损失几乎不下降、准确率不提升的问题。

**关键解决方案：**
1. **扩展网络** - 从3-5层扩展到**可配置深度**（默认7层）
2. **更好的初始化** - Kaiming初始化防止梯度消失/爆炸
3. **批归一化** - 替换InstanceNorm，实现更快更稳定的训练
4. **学习率策略** - 降低学习率(0.0001) + 预热(5轮) + 余弦退火
5. **梯度裁剪** - 防止深度网络中的梯度爆炸
6. **标签平滑** - 提高泛化能力，防止过度自信
7. **残差连接** - 改善深层网络的梯度流动
8. **AdamW优化器** - 带权重衰减的更好正则化

**详见 [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) 获取完整细节。**

### 🚀 快速开始

#### 方式 1：Kaggle 笔记本（最简单）

1. 将 `semg_hht_cnn_classifier.ipynb` 上传到 Kaggle
2. 添加 **HILBERTMATRIX_NPZ** 数据集到笔记本
3. 启用 GPU 加速器（设置 → 加速器 → GPU）
4. 运行所有单元格

笔记本会自动检测 Kaggle 环境并从 `/kaggle/input/hilbertmatrix-npz/hht_matrices/` 加载数据。

#### 方式 2：命令行训练（本地/服务器）

```bash
# 安装依赖
pip install -r requirements.txt

# 使用您的数据训练（新的双分类器系统）
python train.py --data_dir ./data --checkpoint_dir ./checkpoints --epochs 100

# 使用自定义参数和可配置网络深度的高级训练
python train.py \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --num_encoder_layers 5 \
    --base_channels 64 \
    --test_size 0.2

# 使用更深的网络训练（7层，更多特征）
python train.py \
    --data_dir ./data \
    --num_encoder_layers 7 \
    --base_channels 64 \
    --epochs 100

# 使用更浅的网络训练（3层，更快训练）
python train.py \
    --data_dir ./data \
    --num_encoder_layers 3 \
    --base_channels 64 \
    --epochs 100

# 从检查点恢复训练
python train.py --data_dir ./data --checkpoint_dir ./checkpoints --resume

# 运行推理
python inference.py --checkpoint ./checkpoints/final --input ./new_data/
```

**新参数：**
- `--num_encoder_layers`: CNN层数（1-8，默认：5）
  - 更多层 = 更深网络，更好的特征提取，训练更慢
  - 更少层 = 训练更快，内存使用更少
- `--base_channels`: 基础通道数（默认：64）
  - 更高的值 = 更多参数，更好的容量

详细说明请参见 [双分类器系统指南](DUAL_CLASSIFIER_GUIDE.md)。

### 📁 数据格式

**文件命名规范：**
```
肌肉名称_动作类型_性别_编号.npz
```

示例：
- `BICEPS_fatiguetest_M_006.npz` → 男性，完整动作
- `TRICEPS_half_F_012.npz` → 女性，半程动作
- `FOREARM_invalid_M_003.npz` → 男性，无效动作
- `Test1_1_015.npz` → 未标注测试文件（以 "Test" 开头）

**文件内容：**
每个 `.npz` 文件包含一个 256×256 的 HHT 矩阵，使用键 `'hht'` 存储。

### 🔧 模块功能

**1. `train.py` - 生产训练脚本**
- 从目录加载 .npz 文件
- 解析文件名提取性别和动作标签
- 训练 CNN 编码器提取特征
- 在提取的特征上训练 SVM 分类器
- 保存检查点（编码器、缩放器、SVM、元数据）
- 在验证集上评估
- 对测试文件（以 "Test" 开头的文件）运行推理

**2. `inference.py` - 推理脚本**
- 从检查点加载训练好的模型
- 处理单个文件或批量文件
- 输出预测结果和置信度分数
- 将结果保存为 JSON

**3. `generate_sample_data.py` - 数据生成器**
- 创建用于测试的合成 256×256 HHT 矩阵
- 生成正确的文件名格式
- 用于测试流程

**4. Jupyter 笔记本 `semg_hht_cnn_classifier.ipynb`**
- 交互式探索和可视化
- 与 Kaggle 数据集集成
- 分步训练工作流程
- 适合实验

### 📈 训练流程

1. **数据加载**：加载所有 .npz 文件，过滤测试文件（以 "Test" 开头）
2. **特征提取**：CNN 编码器处理 256×256 矩阵 → 256 维向量
3. **归一化**：StandardScaler 归一化特征
4. **SVM 训练**：RBF 核 SVM 在归一化特征上训练
5. **验证**：计算准确率、精确率、召回率、F1 分数
6. **测试推理**：预测测试文件的标签
7. **保存检查点**：保存完整模型状态

---

## 📚 Additional Resources | 其他资源

- **Detailed Training Guide | 详细训练指南**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Scripts Documentation | 脚本文档**: [SCRIPTS_README.md](SCRIPTS_README.md)
- **Example Workflow | 示例工作流**: [example_workflow.sh](example_workflow.sh)

## 📄 License | 许可证

MIT License - See main repository for details.
MIT 许可证 - 详见主仓库。

## 🤝 Contributing | 贡献

Contributions are welcome! Please maintain the CNN architecture and update documentation.
欢迎贡献！请保持 CNN 架构不变并更新文档。
