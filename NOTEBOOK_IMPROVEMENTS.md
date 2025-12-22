# Jupyter Notebook Training Improvements

## 改进说明 | Improvement Summary

本次改进解决了Jupyter Notebook训练过程中的多个问题，并增加了新功能。

This improvement addresses several issues in the Jupyter Notebook training process and adds new features.

## 解决的问题 | Issues Resolved

### 1. 学习率过度衰减 | Excessive Learning Rate Decay

**问题 | Problem:**
- 学习率最小值设置为 `1e-7`，导致训练后期学习率过低，影响模型收敛
- Minimum learning rate was set to `1e-7`, causing extremely low learning rates in later training stages

**解决方案 | Solution:**
- 将 `LR_SCHEDULER_MIN_LR` 从 `1e-7` 改为 `1e-6`
- Changed `LR_SCHEDULER_MIN_LR` from `1e-7` to `1e-6`

**位置 | Location:** Cell 6 - Hyperparameter Configuration

### 2. 多轮训练支持 | Multiple Training Rounds Support

**新功能 | New Feature:**
- 添加了训练轮数配置，支持多轮训练
- Added training rounds configuration to support multiple training rounds

**新增常量 | New Constants:**
```python
NUM_TRAINING_ROUNDS = 3      # 总训练轮数 | Total training rounds
EPOCHS_PER_ROUND = 100       # 每轮训练的epoch数 | Epochs per training round
```

**功能说明 | Features:**
- 每轮训练100个epoch，总共训练3轮（可配置）
- Each round trains for 100 epochs, total 3 rounds (configurable)
- 训练进度显示当前轮次和总轮次
- Training progress shows current round and total rounds
- 检查点文件名包含轮次信息
- Checkpoint filenames include round information

**位置 | Location:** Cell 6 - Hyperparameter Configuration, Cell 12 - Training Function

### 3. 继续训练功能 | Resume Training Support

**新功能 | New Feature:**
- 支持从上次训练的检查点继续训练
- Support resuming training from the last checkpoint

**实现细节 | Implementation Details:**
- 自动检测是否存在之前的检查点 `best_action_quality_model.pt`
- Automatically detects if previous checkpoint `best_action_quality_model.pt` exists
- 如果存在，加载模型状态、优化器状态、训练历史和最佳验证准确率
- If exists, loads model state, optimizer state, training history, and best validation accuracy
- 从上次停止的epoch继续训练
- Continues training from the last stopped epoch

**新增参数 | New Parameters:**
```python
def train_action_quality_model(
    ...
    resume_from=None,           # 检查点路径 | Checkpoint path
    num_rounds=1,               # 训练轮数 | Number of rounds
    epochs_per_round=100        # 每轮epoch数 | Epochs per round
):
```

**位置 | Location:** Cell 12 - Training Function

### 4. DataLoader多进程错误 | DataLoader Multiprocessing Error

**问题 | Problem:**
```
Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__>
IOStream.flush timed out
RuntimeError: cannot join current thread
AssertionError: can only test a child process
```

**原因 | Cause:**
- Jupyter Notebook环境中使用多进程会导致各种问题
- Using multiprocessing in Jupyter Notebook environment causes various issues
- 特别是在notebook的主线程中启动DataLoader workers
- Especially when starting DataLoader workers in the notebook's main thread

**解决方案 | Solution:**
- 将 `num_workers` 从 `2` 改为 `0`
- Changed `num_workers` from `2` to `0`
- 这将使DataLoader在主进程中加载数据，避免多进程问题
- This makes DataLoader load data in the main process, avoiding multiprocessing issues

**位置 | Location:** Cell 12 - Training Function

### 5. torch.load的UnpicklingError | torch.load UnpicklingError

**问题 | Problem:**
```
UnpicklingError: Weights only load failed...
WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default
```

**原因 | Cause:**
- PyTorch 2.6开始，`torch.load`的`weights_only`参数默认值从`False`改为`True`
- Starting from PyTorch 2.6, the default value of `weights_only` parameter in `torch.load` changed from `False` to `True`
- 这导致加载包含numpy对象的检查点时出错
- This causes errors when loading checkpoints containing numpy objects

**解决方案 | Solution:**
- 在所有 `torch.load` 调用中显式添加 `weights_only=False` 参数
- Explicitly add `weights_only=False` parameter to all `torch.load` calls

**修改位置 | Modified Locations:**
- Cell 12: 训练函数中加载检查点 | Loading checkpoint in training function
- Cell 16: SVM训练中加载模型 | Loading model in SVM training

## 使用说明 | Usage Instructions

### 首次训练 | First Training

1. 运行所有单元格直到训练单元格（Cell 12）
2. Run all cells up to the training cell (Cell 12)
3. 训练将自动开始，总共训练300个epoch（3轮 × 100 epoch/轮）
4. Training will automatically start, total 300 epochs (3 rounds × 100 epochs/round)

### 继续训练 | Resume Training

1. 如果之前已经训练过，再次运行训练单元格（Cell 12）
2. If previously trained, run the training cell (Cell 12) again
3. 系统会自动检测并加载之前的检查点
4. The system will automatically detect and load the previous checkpoint
5. 训练将从上次停止的地方继续
6. Training will continue from where it stopped

### 修改训练轮数 | Modify Training Rounds

在 Cell 6 中修改以下常量：
Modify the following constants in Cell 6:

```python
NUM_TRAINING_ROUNDS = 5      # 改为5轮 | Change to 5 rounds
EPOCHS_PER_ROUND = 50        # 每轮50个epoch | 50 epochs per round
```

## 输出示例 | Output Examples

### 新训练 | New Training
```
🆕 开始新的训练 | Starting new training

================================================================================
开始训练动作质量分类器 | Starting Action Quality Classifier Training
================================================================================

🚀 训练配置 | Training configuration:
   设备 Device: cuda
   训练样本 Training samples: 800
   验证样本 Validation samples: 200
   训练轮数 Training rounds: 3
   每轮epoch数 Epochs per round: 100
   总epoch数 Total epochs: 300
   起始epoch Starting epoch: 0
   ...

Round [1/3] Epoch [  1/300] | Train Loss: 0.8234 | Train Acc: 0.6500 | ...
Round [1/3] Epoch [  2/300] | Train Loss: 0.7123 | Train Acc: 0.7100 | ...
...
Round [2/3] Epoch [101/300] | Train Loss: 0.3456 | Train Acc: 0.8900 | ...
...
Round [3/3] Epoch [300/300] | Train Loss: 0.2123 | Train Acc: 0.9300 | ...
```

### 继续训练 | Resume Training
```
♻️  发现检查点，将继续训练 | Found checkpoint, will resume training

📂 从检查点恢复训练 | Resuming training from checkpoint: ./checkpoints/best_action_quality_model.pt
   ✅ 已恢复到epoch 150, 最佳验证准确率: 0.9123
   ✅ Resumed to epoch 150, best val acc: 0.9123

🚀 训练配置 | Training configuration:
   ...
   起始epoch Starting epoch: 150
   ...

Round [2/3] Epoch [150/300] | Train Loss: 0.3012 | Train Acc: 0.9000 | ...
```

## 技术细节 | Technical Details

### 检查点格式 | Checkpoint Format

检查点现在包含以下信息：
Checkpoints now contain the following information:

```python
{
    'epoch': current_epoch,           # 当前epoch | Current epoch
    'round': current_round,           # 当前轮次 | Current round
    'model_state_dict': ...,          # 模型参数 | Model parameters
    'optimizer_state_dict': ...,      # 优化器状态 | Optimizer state
    'best_val_acc': ...,              # 最佳验证准确率 | Best validation accuracy
    'history': {                      # 训练历史 | Training history
        'train_loss': [...],
        'train_acc': [...],
        'val_loss': [...],
        'val_acc': [...],
        'lr': [...]
    }
}
```

### 文件命名 | File Naming

- 最佳模型: `best_action_quality_model.pt`
- Best model: `best_action_quality_model.pt`
- 定期检查点: `action_quality_round_{round}_epoch_{epoch}.pt`
- Periodic checkpoints: `action_quality_round_{round}_epoch_{epoch}.pt`

## 注意事项 | Notes

1. **学习率调度器 | Learning Rate Scheduler:**
   - 使用余弦退火调度器，学习率会在训练过程中逐渐降低
   - Uses cosine annealing scheduler, learning rate gradually decreases during training
   - 最小学习率现在设置为 `1e-6`，避免过度衰减
   - Minimum learning rate now set to `1e-6` to avoid excessive decay

2. **继续训练时的学习率 | Learning Rate When Resuming:**
   - 继续训练时会恢复优化器状态，包括学习率
   - Optimizer state including learning rate is restored when resuming
   - 学习率将从保存时的值继续
   - Learning rate will continue from the saved value

3. **性能考虑 | Performance Considerations:**
   - `num_workers=0` 可能会稍微降低数据加载速度
   - `num_workers=0` may slightly slow down data loading
   - 但在Jupyter环境中这是必要的，以避免多进程问题
   - But this is necessary in Jupyter environment to avoid multiprocessing issues

4. **磁盘空间 | Disk Space:**
   - 定期检查点每10个epoch保存一次
   - Periodic checkpoints are saved every 10 epochs
   - 长时间训练可能会占用较多磁盘空间
   - Long training sessions may consume significant disk space
   - 可以适当增加 `CHECKPOINT_INTERVAL` 值来减少检查点数量
   - Can increase `CHECKPOINT_INTERVAL` value to reduce number of checkpoints
