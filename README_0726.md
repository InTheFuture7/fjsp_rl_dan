# FJSP-DRL 多目标优化项目总结

## 项目概述

本项目基于 [FJSP-DRL](https://github.com/wrqccc/FJSP-DRL) 进行多目标优化改造，实现了柔性作业车间调度问题（FJSP）的双目标优化。项目使用PPO强化学习算法结合双注意力网络（DAN），同时优化两个关键目标：**最小化最大完工时间（Makespan）**和**最小化机器总负载（Total Machine Load）**。

### 原始论文
- **论文标题**: "Flexible Job Shop Scheduling via Dual Attention Network Based Reinforcement Learning"

## 双目标优化功能

### 目标函数定义

**目标1 - 最小化最大完工时间（Makespan）：**
- **定义**：所有工件并行加工时，最后一个工件完成加工的时间点
- **计算方式**：`max(所有工件的完成时间)`
- **优化方向**：越小越好

**目标2 - 最小化机器总负载（Total Machine Load）：**
- **定义**：所有机器的加工时间总和
- **计算方式**：`sum(每台机器的总加工时间)`
- **优化方向**：越小越好

### 多目标优化策略

项目采用**加权线性组合**方法将两个目标合并为单一奖励函数：

```python
# 多目标奖励函数
alpha = 0.7  # 权重系数，可调整
reward = alpha * makespan_reward - (1 - alpha) * machine_total_load
```

- `alpha = 0.7`：更重视makespan优化
- `alpha = 0.5`：两个目标等权重
- `alpha = 0.3`：更重视机器负载优化

## 技术架构

### 核心组件
1. **PPO强化学习算法**：策略优化算法，适合连续决策问题
2. **双注意力网络（DAN）**：
   - 操作注意力块（OAB）：处理工件-操作关系
   - 机器注意力块（MAB）：处理机器-操作关系
3. **多环境并行训练**：支持批量环境同时训练，提高效率

### 环境类设计
- **FJSPEnvForSameOpNums**：处理相同操作数的FJSP实例
- **FJSPEnvForVariousOpNums**：处理不同操作数的FJSP实例
- **统一接口**：两个环境类现在都支持双目标计算

## 使用方法

### 环境要求
```
python == 3.7.11
numpy == 1.21.6
torch == 1.11.0+cu113
ortools == 9.3.10497
pandas == 1.3.5
tqdm == 4.64.0
```

### 训练模型
```bash
# 基础训练（10个工件，5台机器）
python train.py

# 自定义参数训练
python train.py --n_j 10 --n_m 5 --data_source SD2 --model_suffix multi_obj

# 调整多目标权重（需修改train.py中的alpha值）
# alpha = 0.7 (默认，偏重makespan)
# alpha = 0.5 (平衡两个目标)
# alpha = 0.3 (偏重机器负载)
```

### 测试模型
```bash
# 贪婪策略测试
python test_trained_model.py --data_source SD2 --model_source SD2 --test_model 10x5+mix

# 采样策略测试（生成多个解决方案）
python test_trained_model.py --test_mode True --sample_times 100
```

```bash
# params.py 中有关于 gpu 默认值

# 使用GPU（默认）
python train.py

# 强制使用CPU
python train.py --device cpu

# 指定特定GPU
python train.py --device cuda --device_id 1
```


### 测试结果格式
测试结果包含三列数据：
- **第1列**：Makespan（最大完工时间）
- **第2列**：Machine Load（机器总负载）
- **第3列**：Computation Time（计算时间）

## 代码结构

### 主要文件说明
- `fjsp_env_same_op_nums.py`：相同操作数环境类（✅已支持双目标）
- `fjsp_env_various_op_nums.py`：不同操作数环境类（✅已支持双目标）
- `train.py`：训练脚本（✅已支持多目标优化）
- `test_trained_model.py`：测试脚本（✅已支持双目标记录）
- `model/`：神经网络模型实现
- `data/`：数据集和实例文件

### 关键方法
```python
# 环境类新增方法
def calculate_machine_total_load(self):
    """计算机器总负载"""
    return np.sum(self.machine_total_work_time, axis=1)

# step方法新接口
state, makespan_reward, machine_load, done = env.step(actions)
```

## 本次修复内容

### 修复任务总结
1. **✅ 修复fjsp_env_various_op_nums.py的机器总负载计算功能**
   - 添加`machine_total_work_time`属性
   - 实现`calculate_machine_total_load()`方法
   - 修改`step()`方法返回4个值

2. **✅ 修复train.py中验证函数的接口兼容性问题**
   - 更新`validate_envs_with_same_op_nums()`函数
   - 更新`validate_envs_with_various_op_nums()`函数

3. **✅ 修复test_trained_model.py的接口兼容性问题**
   - 更新测试函数的step()调用
   - 确保正确记录双目标数据

4. **✅ 运行测试验证修复效果**
   - 机器负载计算正确性验证通过
   - 接口兼容性验证通过
   - 训练流程验证通过

### 验证结果
- ✅ 机器负载计算100%正确
- ✅ 两个环境类接口完全统一  
- ✅ step方法正确返回4个值
- ✅ 训练和测试流程完整可用
- ✅ 双目标优化功能正常工作

## 未来扩展方向

1. **帕累托前沿分析**：实现真正的多目标优化，生成帕累托最优解集
2. **动态权重调整**：训练过程中自适应调整目标权重
3. **更多目标函数**：添加能耗、设备利用率等其他优化目标
4. **可视化工具**：开发调度结果和目标权衡的可视化界面
5. **基准测试**：在标准FJSP测试集上进行多目标性能评估

## 联系信息

本项目基于原始FJSP-DRL项目进行多目标优化改造。如有问题或建议，请参考原项目文档或提交Issue。

---
**更新日期**: 2024-07-26  
**版本**: 多目标优化版本 v1.0  
**状态**: ✅ 多目标优化功能已完成并验证
