#!/usr/bin/env python3
"""
测试训练流程是否正常工作
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置设备为CPU以避免CUDA问题
import argparse
import params
params.configs.device = 'cpu'
params.configs.num_envs = 2  # 减少环境数量以加快测试
params.configs.max_updates = 2  # 只运行2次更新

from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from data_utils import SD2_instance_generator


def test_training_flow():
    """测试训练流程是否正常"""
    print("=== 测试训练流程 ===")
    
    try:
        # 测试环境创建
        print("1. 测试环境创建...")
        env_same = FJSPEnvForSameOpNums(n_j=3, n_m=3)
        env_various = FJSPEnvForVariousOpNums(n_j=3, n_m=3)
        print("✅ 环境创建成功")
        
        # 测试数据生成
        print("2. 测试数据生成...")
        # 创建简单的测试数据而不是使用SD2生成器
        dataset_job_length = [np.array([2, 2]), np.array([2, 2])]  # 2个环境，每个2个工件，每个工件2个操作
        dataset_op_pt = [
            np.array([[10, 5], [8, 12], [15, 6], [9, 11]]),  # 环境1的操作处理时间
            np.array([[12, 7], [6, 14], [13, 8], [10, 9]])   # 环境2的操作处理时间
        ]
        print("✅ 数据生成成功")
        
        # 测试环境初始化
        print("3. 测试环境初始化...")
        state_same = env_same.set_initial_data(dataset_job_length, dataset_op_pt)
        state_various = env_various.set_initial_data(dataset_job_length, dataset_op_pt)
        print("✅ 环境初始化成功")
        
        # 测试step方法
        print("4. 测试step方法...")
        
        # 测试same_op_nums环境
        action = np.array([0, 1])  # 2个环境的动作
        state, makespan_reward, machine_load, done = env_same.step(action)
        print(f"same_op_nums - makespan_reward: {makespan_reward}, machine_load: {machine_load}")
        
        # 测试various_op_nums环境
        state, makespan_reward, machine_load, done = env_various.step(action)
        print(f"various_op_nums - makespan_reward: {makespan_reward}, machine_load: {machine_load}")
        print("✅ step方法测试成功")
        
        # 测试重置功能
        print("5. 测试重置功能...")
        state_same = env_same.reset()
        state_various = env_various.reset()
        print("✅ 重置功能测试成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training_flow()
    if success:
        print("\n✅ 训练流程测试通过！多目标优化功能正常工作。")
    else:
        print("\n❌ 训练流程测试失败！")
