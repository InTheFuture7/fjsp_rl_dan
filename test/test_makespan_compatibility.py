#!/usr/bin/env python3
"""
测试机器负载修正不影响现有的makespan计算
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置设备为CPU以避免CUDA问题
import argparse
import params
params.configs.device = 'cpu'

from fjsp_env_same_op_nums import FJSPEnvForSameOpNums


def test_makespan_compatibility():
    """测试makespan计算不受机器负载修正影响"""
    print("=== 测试makespan计算兼容性 ===")
    
    # 创建测试环境
    n_j = 2  # 2个工件
    n_m = 2  # 2台机器
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    
    # 创建测试数据
    job_length = np.array([[2, 2]])  # 1个环境，2个工件，每个工件2个操作
    
    # 操作处理时间矩阵
    op_pt = np.array([[[10, 0],   # 操作0: 机器0=10, 机器1=不可用
                       [0, 15],   # 操作1: 机器0=不可用, 机器1=15
                       [20, 0],   # 操作2: 机器0=20, 机器1=不可用
                       [0, 25]]])  # 操作3: 机器0=不可用, 机器1=25
    
    # 初始化环境
    state = env.set_initial_data(job_length, op_pt)
    
    print(f"初始makespan: {env.current_makespan[0]}")
    
    # 执行操作序列
    actions = [
        0 * n_m + 0,  # 工件0的操作0分配给机器0 (时间0-10)
        1 * n_m + 0,  # 工件1的操作2分配给机器0 (时间10-30)
        0 * n_m + 1,  # 工件0的操作1分配给机器1 (时间0-15)
        1 * n_m + 1   # 工件1的操作3分配给机器1 (时间15-40)
    ]
    
    makespans = []
    for i, action in enumerate(actions):
        print(f"\n--- 执行动作 {i+1}: {action} ---")
        state, reward, machine_load, done = env.step(np.array([action]))
        current_makespan = env.current_makespan[0]
        makespans.append(current_makespan)
        print(f"当前makespan: {current_makespan}")
        print(f"当前机器总负载: {machine_load[0]}")
        
        if done.all():
            break
    
    # 验证makespan计算
    # 根据我们的调度：
    # 机器0: 操作0(0-10) -> 操作2(10-30)，完成时间30
    # 机器1: 操作1(0-15) -> 操作3(15-40)，完成时间40
    # makespan应该是max(30, 40) = 40
    expected_makespan = 40
    final_makespan = env.current_makespan[0]
    
    print(f"\n=== makespan验证结果 ===")
    print(f"期望makespan: {expected_makespan}")
    print(f"实际makespan: {final_makespan}")
    print(f"makespan正确: {final_makespan == expected_makespan}")
    
    # 验证makespan是单调递增的
    makespan_monotonic = all(makespans[i] >= makespans[i-1] for i in range(1, len(makespans)))
    print(f"makespan单调递增: {makespan_monotonic}")
    
    return final_makespan == expected_makespan and makespan_monotonic


if __name__ == "__main__":
    success = test_makespan_compatibility()
    if success:
        print("\n✅ makespan计算兼容性测试通过！")
        sys.exit(0)
    else:
        print("\n❌ makespan计算兼容性测试失败！")
        sys.exit(1)
