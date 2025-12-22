# Configurable Architecture Guide | 可配置架构指南

[English](#english) | [中文](#chinese)

---

## <a name="english"></a>English Version

### Overview

This guide explains the configurable CNN architecture feature added to the sEMG-HHT classification system. The network depth can now be configured as a hyperparameter, allowing you to choose the optimal balance between model capacity and training efficiency.

### What Changed?

#### 1. Configurable Network Depth

**Before:** The network had a fixed number of layers (7 layers hardcoded).

**Now:** You can configure the number of convolutional layers from 1 to 8.

```python
# In train.py
python train.py --data_dir ./data --num_encoder_layers 5

# In notebook
MODEL_NUM_LAYERS = 7  # Can be changed from 1 to 8
```

#### 2. Adaptive Classifier Head

**Before:** Fixed classifier head dimensions (2048 → 1024 → 512 → 3).

**Now:** The classifier head automatically scales based on the encoder's output dimension:

```python
# For 3-layer encoder (256-dim features):
Classifier: 256 → 256 → 128 → 3

# For 5-layer encoder (1024-dim features):
Classifier: 1024 → 512 → 256 → 3

# For 7-layer encoder (2048-dim features):
Classifier: 2048 → 1024 → 512 → 3
```

#### 3. Input Validation

The network now validates input parameters to prevent common errors:

```python
# These will raise clear error messages:
num_layers = 0    # Error: num_layers must be at least 1
num_layers = 9    # Error: num_layers must be at most 8 for 256x256 input
```

#### 4. Safe Checkpoint Saving

**Before:** Direct `torch.save()` could fail if disk is full, corrupting the checkpoint.

**Now:** 
- Checks available disk space before saving
- Uses atomic writes (temporary file + rename)
- Automatically cleans up on failure

```python
# Safe checkpoint saving with error handling
try:
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, final_path)  # Atomic operation
except Exception as e:
    print(f"Failed to save: {e}")
    cleanup_temp_file()
```

### How to Use

#### Command Line (train.py)

```bash
# Default configuration (5 layers)
python train.py --data_dir ./data

# Shallow network (faster training, less memory)
python train.py --data_dir ./data --num_encoder_layers 3 --batch_size 32

# Medium network (balanced)
python train.py --data_dir ./data --num_encoder_layers 5 --batch_size 16

# Deep network (best features, slower training)
python train.py --data_dir ./data --num_encoder_layers 7 --batch_size 8

# Very deep network (maximum capacity)
python train.py --data_dir ./data --num_encoder_layers 8 --batch_size 4

# Custom base channels (more/less capacity at each layer)
python train.py --data_dir ./data --num_encoder_layers 5 --base_channels 32
```

#### Jupyter Notebook

Simply change the hyperparameter at the top of the notebook:

```python
# Hyperparameter Configuration Cell
MODEL_NUM_LAYERS = 5  # Change this from 1 to 8
```

The model will automatically adapt to your configuration.

### Performance Comparison

| Layers | Feature Dim | Parameters | Memory (GB) | Speed | Recommended Use |
|--------|-------------|------------|-------------|-------|-----------------|
| 3      | 256         | ~2M        | ~2         | Fast  | Quick experiments, limited data |
| 5      | 1024        | ~8M        | ~4         | Medium| Default, balanced performance |
| 7      | 2048        | ~20M       | ~8         | Slow  | Best accuracy, large dataset |
| 8      | 2048        | ~22M       | ~10        | Slower| Maximum capacity |

*Note: Memory and speed estimates for batch_size=16 with 256×256 input*

### Best Practices

1. **Start with default (5 layers)** - Good balance for most cases
2. **Use shallow networks (3 layers)** when:
   - Limited training data (< 500 samples)
   - Limited GPU memory
   - Need fast training/inference
3. **Use deep networks (7-8 layers)** when:
   - Large training dataset (> 2000 samples)
   - GPU with >= 8GB memory available
   - Accuracy is more important than speed

4. **Adjust batch size** based on network depth:
   - 3 layers: batch_size 32-64
   - 5 layers: batch_size 16-32
   - 7 layers: batch_size 8-16
   - 8 layers: batch_size 4-8

### Troubleshooting

**Problem:** Out of memory error during training

**Solution:** 
```bash
# Reduce layers or batch size
python train.py --data_dir ./data --num_encoder_layers 3 --batch_size 8
```

**Problem:** Disk write error when saving checkpoint

**Solution:** The new safe checkpoint saving will automatically detect this and skip saving. Free up disk space (>500MB recommended) and the next checkpoint will save successfully.

**Problem:** Training too slow

**Solution:**
```bash
# Use fewer layers
python train.py --data_dir ./data --num_encoder_layers 3
```

**Problem:** Poor accuracy with shallow network

**Solution:**
```bash
# Increase network depth
python train.py --data_dir ./data --num_encoder_layers 7 --epochs 150
```

---

## <a name="chinese"></a>中文版本

### 概述

本指南说明了添加到 sEMG-HHT 分类系统的可配置 CNN 架构功能。网络深度现在可以配置为超参数，让您能够选择模型容量和训练效率之间的最佳平衡。

### 有什么变化？

#### 1. 可配置的网络深度

**之前：** 网络具有固定数量的层（硬编码为7层）。

**现在：** 您可以配置卷积层的数量，范围从1到8层。

```python
# 在 train.py 中
python train.py --data_dir ./data --num_encoder_layers 5

# 在笔记本中
MODEL_NUM_LAYERS = 7  # 可以从1到8更改
```

#### 2. 自适应分类头

**之前：** 固定的分类头维度（2048 → 1024 → 512 → 3）。

**现在：** 分类头根据编码器的输出维度自动调整：

```python
# 对于3层编码器（256维特征）：
分类器: 256 → 256 → 128 → 3

# 对于5层编码器（1024维特征）：
分类器: 1024 → 512 → 256 → 3

# 对于7层编码器（2048维特征）：
分类器: 2048 → 1024 → 512 → 3
```

#### 3. 输入验证

网络现在验证输入参数以防止常见错误：

```python
# 这些将引发清晰的错误消息：
num_layers = 0    # 错误：num_layers 必须至少为1
num_layers = 9    # 错误：对于256x256输入，num_layers 最多为8
```

#### 4. 安全的检查点保存

**之前：** 直接使用 `torch.save()` 如果磁盘已满可能失败，损坏检查点。

**现在：**
- 保存前检查可用磁盘空间
- 使用原子写入（临时文件 + 重命名）
- 失败时自动清理

```python
# 带错误处理的安全检查点保存
try:
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, final_path)  # 原子操作
except Exception as e:
    print(f"保存失败: {e}")
    cleanup_temp_file()
```

### 如何使用

#### 命令行 (train.py)

```bash
# 默认配置（5层）
python train.py --data_dir ./data

# 浅层网络（训练更快，内存更少）
python train.py --data_dir ./data --num_encoder_layers 3 --batch_size 32

# 中等网络（平衡）
python train.py --data_dir ./data --num_encoder_layers 5 --batch_size 16

# 深层网络（最佳特征，训练较慢）
python train.py --data_dir ./data --num_encoder_layers 7 --batch_size 8

# 非常深的网络（最大容量）
python train.py --data_dir ./data --num_encoder_layers 8 --batch_size 4

# 自定义基础通道数（每层更多/更少容量）
python train.py --data_dir ./data --num_encoder_layers 5 --base_channels 32
```

#### Jupyter 笔记本

只需在笔记本顶部更改超参数：

```python
# 超参数配置单元格
MODEL_NUM_LAYERS = 5  # 将此值从1更改为8
```

模型将自动适应您的配置。

### 性能比较

| 层数 | 特征维度 | 参数量 | 内存(GB) | 速度 | 推荐用途 |
|------|----------|--------|----------|------|----------|
| 3    | 256      | ~2M    | ~2      | 快   | 快速实验，有限数据 |
| 5    | 1024     | ~8M    | ~4      | 中等 | 默认，平衡性能 |
| 7    | 2048     | ~20M   | ~8      | 慢   | 最佳准确率，大数据集 |
| 8    | 2048     | ~22M   | ~10     | 更慢 | 最大容量 |

*注意：内存和速度估计基于batch_size=16，256×256输入*

### 最佳实践

1. **从默认配置开始（5层）** - 大多数情况下的良好平衡
2. **使用浅层网络（3层）** 当：
   - 训练数据有限（< 500个样本）
   - GPU内存有限
   - 需要快速训练/推理
3. **使用深层网络（7-8层）** 当：
   - 大型训练数据集（> 2000个样本）
   - GPU具有 >= 8GB可用内存
   - 准确性比速度更重要

4. **根据网络深度调整批次大小**：
   - 3层：batch_size 32-64
   - 5层：batch_size 16-32
   - 7层：batch_size 8-16
   - 8层：batch_size 4-8

### 故障排除

**问题：** 训练期间出现内存不足错误

**解决方案：**
```bash
# 减少层数或批次大小
python train.py --data_dir ./data --num_encoder_layers 3 --batch_size 8
```

**问题：** 保存检查点时出现磁盘写入错误

**解决方案：** 新的安全检查点保存将自动检测此问题并跳过保存。释放磁盘空间（建议 >500MB），下一个检查点将成功保存。

**问题：** 训练太慢

**解决方案：**
```bash
# 使用更少的层
python train.py --data_dir ./data --num_encoder_layers 3
```

**问题：** 浅层网络准确率不佳

**解决方案：**
```bash
# 增加网络深度
python train.py --data_dir ./data --num_encoder_layers 7 --epochs 150
```
