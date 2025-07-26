#!/usr/bin/env python3
"""
简单测试验证机器负载计算修正的正确性
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


def test_machine_load_calculation():
    """测试机器负载计算的正确性"""
    print("=== 测试机器负载计算修正 ===")
    
    # 创建简单的测试环境
    n_j = 2  # 2个工件
    n_m = 2  # 2台机器
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    
    # 创建简单的测试数据
    # 工件长度：每个工件2个操作
    job_length = np.array([[2, 2]])  # 1个环境，2个工件，每个工件2个操作
    
    # 操作处理时间矩阵 [1环境, 4操作, 2机器]
    # 操作0: 工件0的第1个操作，可在机器0(时间10)或机器1(时间0-不可用)
    # 操作1: 工件0的第2个操作，可在机器0(时间0-不可用)或机器1(时间15)
    # 操作2: 工件1的第1个操作，可在机器0(时间20)或机器1(时间0-不可用)
    # 操作3: 工件1的第2个操作，可在机器0(时间0-不可用)或机器1(时间25)
    op_pt = np.array([[[10, 0],   # 操作0: 机器0=10, 机器1=不可用
                       [0, 15],   # 操作1: 机器0=不可用, 机器1=15
                       [20, 0],   # 操作2: 机器0=20, 机器1=不可用
                       [0, 25]]])  # 操作3: 机器0=不可用, 机器1=25
    
    # 初始化环境
    state = env.set_initial_data(job_length, op_pt)
    
    print(f"初始机器工作时间累计器: {env.machine_total_work_time}")
    
    # 手工计算期望的机器负载
    # 如果按照操作0->操作2->操作1->操作3的顺序执行：
    # 机器0: 操作0(10) + 操作2(20) = 30
    # 机器1: 操作1(15) + 操作3(25) = 40
    # 总负载: 30 + 40 = 70
    expected_machine_0_load = 10 + 20  # 操作0 + 操作2
    expected_machine_1_load = 15 + 25  # 操作1 + 操作3
    expected_total_load = expected_machine_0_load + expected_machine_1_load
    
    print(f"期望的机器0负载: {expected_machine_0_load}")
    print(f"期望的机器1负载: {expected_machine_1_load}")
    print(f"期望的总负载: {expected_total_load}")
    
    # 模拟执行操作序列
    actions = [
        0 * n_m + 0,  # 工件0的操作0分配给机器0 (action = job*n_m + machine)
        1 * n_m + 0,  # 工件1的操作2分配给机器0
        0 * n_m + 1,  # 工件0的操作1分配给机器1
        1 * n_m + 1   # 工件1的操作3分配给机器1
    ]
    
    for i, action in enumerate(actions):
        print(f"\n--- 执行动作 {i+1}: {action} ---")
        state, reward, machine_load, done = env.step(np.array([action]))
        print(f"当前机器工作时间: {env.machine_total_work_time[0]}")
        print(f"当前机器总负载: {machine_load[0]}")
        
        if done.all():
            break
    
    # 验证最终结果
    final_machine_load = env.calculate_machine_total_load()
    print(f"\n=== 最终结果 ===")
    print(f"最终机器工作时间: {env.machine_total_work_time[0]}")
    print(f"最终机器总负载: {final_machine_load[0]}")
    print(f"期望总负载: {expected_total_load}")
    
    # 验证计算正确性
    machine_0_actual = env.machine_total_work_time[0, 0]
    machine_1_actual = env.machine_total_work_time[0, 1]
    total_actual = final_machine_load[0]
    
    print(f"\n=== 验证结果 ===")
    print(f"机器0负载 - 期望: {expected_machine_0_load}, 实际: {machine_0_actual}, 正确: {machine_0_actual == expected_machine_0_load}")
    print(f"机器1负载 - 期望: {expected_machine_1_load}, 实际: {machine_1_actual}, 正确: {machine_1_actual == expected_machine_1_load}")
    print(f"总负载 - 期望: {expected_total_load}, 实际: {total_actual}, 正确: {total_actual == expected_total_load}")
    
    # 测试重置功能
    print(f"\n=== 测试重置功能 ===")
    print(f"重置前机器工作时间: {env.machine_total_work_time[0]}")
    env.reset()
    print(f"重置后机器工作时间: {env.machine_total_work_time[0]}")
    reset_correct = np.all(env.machine_total_work_time[0] == 0)
    print(f"重置正确: {reset_correct}")
    
    return (machine_0_actual == expected_machine_0_load and 
            machine_1_actual == expected_machine_1_load and 
            total_actual == expected_total_load and 
            reset_correct)


if __name__ == "__main__":
    success = test_machine_load_calculation()
    if success:
        print("\n✅ 所有测试通过！机器负载计算修正成功。")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！需要检查实现。")
        sys.exit(1)
