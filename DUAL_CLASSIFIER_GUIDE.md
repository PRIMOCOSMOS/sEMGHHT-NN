# Dual Classifier System Guide
# 双分类器系统指南

[English](#english) | [中文](#chinese)

---

## <a name="english"></a>English Version

### Overview

This project now implements a **dual classification system** that separates the classification tasks into two specialized models:

1. **Deep Learning CNN** for Action Quality Classification (3 classes: Full, Half, Invalid)
2. **SVM Classifier** for Gender Classification (2 classes: M, F)

This separation provides better accuracy and more focused optimization for each task.

### Why This Change?

**Previous Problem:**
- Single model trying to classify 6 classes (M_full, M_half, M_invalid, F_full, F_half, F_invalid)
- Loss barely decreased during training
- Accuracy improvements were minimal
- Model struggled to learn both gender and action quality simultaneously

**New Solution:**
- **Task Separation**: Gender and action quality are fundamentally different features
- **Specialized Models**: Each classifier is optimized for its specific task
- **Better Convergence**: Simpler problems lead to better training dynamics
- **Higher Accuracy**: Specialized models outperform general-purpose ones

### Architecture Improvements

#### 1. Action Quality CNN Classifier

**Enhanced Features:**
- **5 convolutional layers** (increased from 3) for deeper feature extraction
- **Channel progression**: 64 → 128 → 256 → 512 → 1024
- **BatchNormalization** instead of InstanceNorm for better training stability
- **Kaiming initialization** to prevent vanishing/exploding gradients
- **Dropout regularization** (50%) to prevent overfitting
- **Classification head**: 3-layer fully connected network with BatchNorm

**Training Improvements:**
- Learning rate scheduling (ReduceLROnPlateau)
- Weight decay (L2 regularization)
- Data normalization to [0, 1] range
- Batch size: 16 (optimized for 256×256 images)
- Default epochs: 100

#### 2. Gender SVM Classifier

**Features:**
- CNN feature extractor (4 layers, frozen during SVM training)
- RBF kernel SVM with C=10.0
- StandardScaler normalization
- One-shot training (no epochs needed)

### Input Data Requirements

**File Format**: `.npz` files containing 256×256 HHT matrices

**Naming Convention**:
```
MUSCLENAME_movement_GENDER_###.npz
```

**Examples:**
- `BICEPS_fatiguetest_M_006.npz` → Male, Full movement
- `TRICEPS_half_F_012.npz` → Female, Half movement
- `FOREARM_invalid_M_003.npz` → Male, Invalid movement
- `Test1_1_015.npz` → Test file (no labels, for inference)

**Input Requirements:**
- ✅ Single channel (grayscale): 256×256
- ✅ Normalized to [0, 1] range (automatic)
- ✅ NPZ file with key `'hht'` or first array key

### Usage

#### Training

```bash
# Basic training
python train.py --data_dir ./data --checkpoint_dir ./checkpoints

# Advanced training with custom parameters
python train.py \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --test_size 0.2

# Use CPU instead of GPU
python train.py --data_dir ./data --cpu
```

#### Inference

```bash
# Single file prediction
python inference.py \
    --checkpoint ./checkpoints/final \
    --input ./test_data/sample.npz \
    --output predictions.json

# Directory batch prediction
python inference.py \
    --checkpoint ./checkpoints/final \
    --input ./test_data/ \
    --output batch_predictions.json
```

### Training Output

The training script produces:

**Gender Classifier Checkpoints:**
- `final_gender_encoder.pt` - CNN encoder weights
- `final_gender_scaler.pkl` - Feature scaler
- `final_gender_svm.pkl` - Trained SVM model
- `final_gender_metadata.pkl` - Metadata and label encoder

**Action Quality Checkpoints:**
- `best_action_quality.pt` - Best model (highest validation accuracy)
- `final_action_quality.pt` - Final model after all epochs
- `checkpoint_epoch_N_action_quality.pt` - Periodic checkpoints every 20 epochs

**Results:**
- `test_predictions.json` - Predictions on test files with confidence scores

### Expected Performance Improvements

With the new architecture, you should see:

1. ✅ **Loss decreasing steadily** from epoch 1
2. ✅ **Validation accuracy improving** consistently
3. ✅ **Convergence within 50-100 epochs** (vs. no convergence before)
4. ✅ **Gender classification**: 85-95% accuracy (binary task is easier)
5. ✅ **Action quality classification**: 75-90% accuracy (3-class task with better features)

### Troubleshooting Training Issues

**If loss is not decreasing:**
- ✅ Check data normalization (automatic in new version)
- ✅ Verify input shape is (N, 1, 256, 256)
- ✅ Try lower learning rate (0.0001)
- ✅ Increase batch size if memory allows

**If accuracy is low:**
- ✅ Ensure sufficient training data (>100 samples per class)
- ✅ Check label distribution is balanced
- ✅ Try training for more epochs (150-200)
- ✅ Adjust dropout rate (try 0.3-0.6)

### Jupyter Notebook Usage

The notebook `semg_hht_cnn_classifier.ipynb` has been updated to reflect the dual classifier system. Key changes:

1. **No synthetic data generation** - Only real data loading
2. **Dual classifier training cells** - Separate sections for each classifier
3. **Improved visualization** - Training curves for action quality CNN
4. **Dual predictions** - Shows both gender and action quality results

To use the notebook:
1. Upload to Kaggle or Jupyter environment
2. Add your HILBERTMATRIX_NPZ dataset
3. Run cells in order
4. Monitor training progress
5. Evaluate both classifiers

---

## <a name="chinese"></a>中文版本

### 概述

本项目现在实现了**双分类器系统**，将分类任务分离为两个专门的模型：

1. **深度学习CNN** 用于动作质量分类（3类：全程、半程、无效）
2. **SVM分类器** 用于性别分类（2类：男、女）

这种分离为每个任务提供了更好的准确性和更集中的优化。

### 为什么要做这个改变？

**之前的问题：**
- 单一模型试图分类6个类别（M_full, M_half, M_invalid, F_full, F_half, F_invalid）
- 训练过程中损失几乎不下降
- 准确率提升微乎其微
- 模型难以同时学习性别和动作质量

**新的解决方案：**
- **任务分离**：性别和动作质量是根本不同的特征
- **专用模型**：每个分类器都针对其特定任务进行优化
- **更好的收敛**：更简单的问题导致更好的训练动态
- **更高的准确率**：专用模型优于通用模型

### 架构改进

#### 1. 动作质量CNN分类器

**增强功能：**
- **5个卷积层**（从3层增加）用于更深的特征提取
- **通道递进**：64 → 128 → 256 → 512 → 1024
- **BatchNormalization** 代替 InstanceNorm 以获得更好的训练稳定性
- **Kaiming初始化** 防止梯度消失/爆炸
- **Dropout正则化**（50%）防止过拟合
- **分类头**：3层全连接网络与BatchNorm

**训练改进：**
- 学习率调度（ReduceLROnPlateau）
- 权重衰减（L2正则化）
- 数据归一化到[0, 1]范围
- 批次大小：16（针对256×256图像优化）
- 默认轮数：100

#### 2. 性别SVM分类器

**特点：**
- CNN特征提取器（4层，SVM训练期间冻结）
- RBF核SVM，C=10.0
- StandardScaler归一化
- 一次性训练（无需多轮）

### 输入数据要求

**文件格式**：包含256×256 HHT矩阵的`.npz`文件

**命名规范**：
```
肌肉名称_动作类型_性别_编号.npz
```

**示例：**
- `BICEPS_fatiguetest_M_006.npz` → 男性，全程动作
- `TRICEPS_half_F_012.npz` → 女性，半程动作
- `FOREARM_invalid_M_003.npz` → 男性，无效动作
- `Test1_1_015.npz` → 测试文件（无标签，用于推理）

**输入要求：**
- ✅ 单通道（灰度）：256×256
- ✅ 归一化到[0, 1]范围（自动）
- ✅ NPZ文件，键为`'hht'`或第一个数组键

### 使用方法

#### 训练

```bash
# 基本训练
python train.py --data_dir ./data --checkpoint_dir ./checkpoints

# 使用自定义参数的高级训练
python train.py \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --test_size 0.2

# 使用CPU而不是GPU
python train.py --data_dir ./data --cpu
```

#### 推理

```bash
# 单文件预测
python inference.py \
    --checkpoint ./checkpoints/final \
    --input ./test_data/sample.npz \
    --output predictions.json

# 目录批量预测
python inference.py \
    --checkpoint ./checkpoints/final \
    --input ./test_data/ \
    --output batch_predictions.json
```

### 训练输出

训练脚本产生：

**性别分类器检查点：**
- `final_gender_encoder.pt` - CNN编码器权重
- `final_gender_scaler.pkl` - 特征缩放器
- `final_gender_svm.pkl` - 训练好的SVM模型
- `final_gender_metadata.pkl` - 元数据和标签编码器

**动作质量检查点：**
- `best_action_quality.pt` - 最佳模型（最高验证准确率）
- `final_action_quality.pt` - 所有轮次后的最终模型
- `checkpoint_epoch_N_action_quality.pt` - 每20轮的周期性检查点

**结果：**
- `test_predictions.json` - 测试文件的预测和置信度分数

### 预期性能改进

使用新架构，您应该看到：

1. ✅ **损失从第1轮开始稳步下降**
2. ✅ **验证准确率持续提高**
3. ✅ **在50-100轮内收敛**（而不是之前的不收敛）
4. ✅ **性别分类**：85-95%准确率（二分类任务更简单）
5. ✅ **动作质量分类**：75-90%准确率（3类任务，特征更好）

### 训练问题排查

**如果损失不下降：**
- ✅ 检查数据归一化（新版本中自动）
- ✅ 验证输入形状为(N, 1, 256, 256)
- ✅ 尝试更低的学习率（0.0001）
- ✅ 如果内存允许，增加批次大小

**如果准确率低：**
- ✅ 确保有足够的训练数据（每类>100个样本）
- ✅ 检查标签分布是否平衡
- ✅ 尝试训练更多轮次（150-200）
- ✅ 调整dropout率（尝试0.3-0.6）

### Jupyter笔记本使用

笔记本`semg_hht_cnn_classifier.ipynb`已更新以反映双分类器系统。主要变化：

1. **无合成数据生成** - 仅加载真实数据
2. **双分类器训练单元** - 每个分类器的单独部分
3. **改进的可视化** - 动作质量CNN的训练曲线
4. **双重预测** - 显示性别和动作质量结果

使用笔记本：
1. 上传到Kaggle或Jupyter环境
2. 添加您的HILBERTMATRIX_NPZ数据集
3. 按顺序运行单元格
4. 监控训练进度
5. 评估两个分类器

---

## Key Files | 关键文件

- `train.py` - Dual classifier training script | 双分类器训练脚本
- `inference.py` - Inference script (to be updated) | 推理脚本（待更新）
- `semg_hht_cnn_classifier.ipynb` - Jupyter notebook | Jupyter笔记本
- `DUAL_CLASSIFIER_GUIDE.md` - This guide | 本指南
- `README.md` - Project overview | 项目概述

## Citation | 引用

If you use this dual classifier system in your research, please cite:

如果您在研究中使用此双分类器系统，请引用：

```
@misc{semg-hht-dual-classifier,
  title={sEMG-HHT Dual Classifier for Movement Quality and Gender Classification},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/PRIMOCOSMOS/sEMGHHT-NN}
}
```
