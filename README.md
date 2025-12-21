# sEMG-HHT CNN Classifier | sEMG-HHT CNN 分类器

[English](#english) | [中文](#chinese)

---

## <a name="english"></a>English Version

A Convolutional Neural Network (CNN) based classifier for surface electromyography (sEMG) signals using Hilbert-Huang Transform (HHT) representation. This project is designed for multi-class classification tasks such as movement quality assessment and gender classification.

### 🎯 Overview

This project implements a deep learning pipeline that:
1. Takes 256×256 HHT matrices as input (derived from sEMG signals)
2. Extracts features using a 3-layer CNN encoder
3. Performs multi-class classification using SVM or end-to-end neural network

### 🏗️ Model Architecture

**CNN Encoder Structure:**
- **3 Convolutional Layers**, each containing:
  - Conv2D (kernel=3, stride=2, padding=1)
  - Instance Normalization (maintains data distribution per sample)
  - LeakyReLU activation (slope=0.2)
- **Global Average Pooling** at the end
- Output: 256-dimensional feature vector

**Classifier Options:**
1. **CNN-SVM (Recommended)**: CNN extracts features → SVM classifies (supports 6-class classification)
2. **End-to-End**: Fully trainable neural network with FC layers

### 📊 Classification Task

**6-Class Multi-Dimensional Classification:**
- **Gender Dimension**: Male (M) / Female (F)
- **Movement Quality Dimension**: 
  - Full (完整动作): Complete movement range
  - Half (半程动作): Partial movement range  
  - Invalid (无效动作): Incorrect or failed movement

**Class Mapping:**
| Class ID | Label | Gender | Movement |
|----------|-------|--------|----------|
| 0 | M_full | Male | Full |
| 1 | M_half | Male | Half |
| 2 | M_invalid | Male | Invalid |
| 3 | F_full | Female | Full |
| 4 | F_half | Female | Half |
| 5 | F_invalid | Female | Invalid |

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

# Train with your data
python train.py --data_dir ./data --checkpoint_dir ./checkpoints

# Resume from checkpoint
python train.py --data_dir ./data --checkpoint_dir ./checkpoints --resume

# Run inference
python inference.py --checkpoint ./checkpoints/final --input ./new_data/
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.

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

基于卷积神经网络（CNN）的表面肌电信号（sEMG）分类器，使用希尔伯特-黄变换（HHT）表示。该项目设计用于动作质量评估和性别分类等多类分类任务。

### 🎯 概述

该项目实现了一个深度学习流程：
1. 输入 256×256 的 HHT 矩阵（从 sEMG 信号导出）
2. 使用 3 层 CNN 编码器提取特征
3. 使用 SVM 或端到端神经网络进行多类分类

### 🏗️ 模型架构

**CNN 编码器结构：**
- **3 个卷积层**，每层包含：
  - Conv2D（kernel=3, stride=2, padding=1）
  - 实例归一化（Instance Normalization，保持每个样本的数据分布）
  - LeakyReLU 激活函数（slope=0.2）
- 末尾使用**全局平均池化**
- 输出：256 维特征向量

**分类器选项：**
1. **CNN-SVM（推荐）**：CNN 提取特征 → SVM 分类（支持 6 类分类）
2. **端到端模型**：全连接层的完全可训练神经网络

### 📊 分类任务

**6 类多维分类：**
- **性别维度**：男性 (M) / 女性 (F)
- **动作质量维度**：
  - Full（完整动作）：完整的运动范围
  - Half（半程动作）：部分运动范围
  - Invalid（无效动作）：错误或失败的动作

**类别映射：**
| 类别 ID | 标签 | 性别 | 动作 |
|---------|------|------|------|
| 0 | M_full | 男性 | 完整 |
| 1 | M_half | 男性 | 半程 |
| 2 | M_invalid | 男性 | 无效 |
| 3 | F_full | 女性 | 完整 |
| 4 | F_half | 女性 | 半程 |
| 5 | F_invalid | 女性 | 无效 |

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

# 使用您的数据训练
python train.py --data_dir ./data --checkpoint_dir ./checkpoints

# 从检查点恢复训练
python train.py --data_dir ./data --checkpoint_dir ./checkpoints --resume

# 运行推理
python inference.py --checkpoint ./checkpoints/final --input ./new_data/
```

详细说明请参见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。

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
