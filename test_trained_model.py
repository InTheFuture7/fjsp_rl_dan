import time
import os
from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import pack_data_from_config
from model.PPO import PPO_initialize
from common_utils import setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)

ppo = PPO_initialize()
test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def test_greedy_strategy(data_set, model_path, seed):
    """
        test the model on the given data using the greedy strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan, machine load, and time
    贪婪策略在每一步选择当前最优的动作
    """

    # 初始化测试结果列表
    test_result_list = []
    # 设置随机种子以确保可重复性
    setup_seed(seed)
    # 加载预训练模型
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    # 将模型设置为评估模式
    ppo.policy.eval()
    # 获取作业数量和机器数量
    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    # 导入FJSP环境
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    # 创建FJSP环境实例
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    # 遍历测试数据集
    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        # 设置初始状态
        state = env.set_initial_data([data_set[0][i]], [data_set[1][i]])
        # 记录开始时间
        t1 = time.time()
        # 贪婪策略循环
        while True:
            # 使用模型生成动作概率分布
            with torch.no_grad():
                pi, _ = ppo.policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor, fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor, dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
            # 贪婪选择动作
            action = greedy_select_action(pi)
            # 执行动作并获取新状态
            state, reward, done = env.step(actions=action.cpu().numpy())
            # 如果调度完成，退出循环
            if done:
                break
        # 记录结束时间
        t2 = time.time()
        # 计算机器总负载
        machine_total_load = env.calculate_machine_total_load()
        # 记录测试结果：完工时间、机器总负载和计算时间
        test_result_list.append([env.current_makespan[0], machine_total_load[0], t2 - t1])
    # 返回测试结果数组
    return np.array(test_result_list)


def test_sampling_strategy(data_set, model_path, sample_times, seed):
    """
        test the model on the given data using the sampling strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan, machine load, and time
    采样策略在每个测试实例上生成多个解决方案，并选择最优的方案
    """
    setup_seed(seed)
    test_result_list = []
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    env = FJSPEnvForSameOpNums(n_j, n_m)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        JobLength_dataset = np.tile(np.expand_dims(data_set[0][i], axis=0), (sample_times, 1))
        OpPT_dataset = np.tile(np.expand_dims(data_set[1][i], axis=0), (sample_times, 1, 1))

        state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
        t1 = time.time()
        best_makespan = float('inf')
        best_machine_load = float('inf')

        for _ in range(sample_times):
            while True:
                with torch.no_grad():
                    pi, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [100, N, 8]
                                       op_mask=state.op_mask_tensor,  # [100, N, N]
                                       candidate=state.candidate_tensor,  # [100, J]
                                       fea_m=state.fea_m_tensor,  # [100, M, 6]
                                       mch_mask=state.mch_mask_tensor,  # [100, M, M]
                                       comp_idx=state.comp_idx_tensor,  # [100, M, M, J]
                                       dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [100, J, M]
                                       fea_pairs=state.fea_pairs_tensor)  # [100, J, M]

                action_envs, _ = sample_action(pi)
                state, _, done = env.step(action_envs.cpu().numpy())
                if done.all():
                    break

            machine_total_load = env.calculate_machine_total_load()
            current_makespan = np.min(env.current_makespan)

            if current_makespan < best_makespan:
                best_makespan = current_makespan
            if machine_total_load < best_machine_load:
                best_machine_load = machine_total_load

        t2 = time.time()
        test_result_list.append([best_makespan, best_machine_load, t2 - t1])

    return np.array(test_result_list)


def main(config, flag_sample):
    """
        test the trained model following the config and save the results
    :param flag_sample: whether using the sampling strategy
    """
    setup_seed(config.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    # collect the path of test models
    test_model = [(f'./trained_network/{config.model_source}/{model_name}.pth', model_name) for model_name in config.test_model]

    # collect the test data
    test_data = pack_data_from_config(config.data_source, config.test_data)

    model_prefix = "DANIELS" if flag_sample else "DANIELG"

    for data in test_data:
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"test mode: {model_prefix}")
        save_direc = f'./test_results/{config.data_source}/{data[1]}'
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model in test_model:
            save_path = save_direc + f'/Result_{model_prefix}+{model[1]}_{data[1]}.npy'
            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{config.data_source}/{data[1]}")

                if not flag_sample:
                    print("Test mode: Greedy")
                    # 运行了5次test_greedy_strategy函数，并将结果存储在result_5_times列表中。
                    # 这样做的目的是为了减少单次评估的随机性对结果的影响，通过多次评估取平均值来获得更稳定的结果。
                    result_5_times = [test_greedy_strategy(data[0], model[0], config.seed_test) for _ in range(5)]
                    save_result = np.mean(result_5_times, axis=0)
                    print("testing results:")
                    print(f"makespan(greedy): ", save_result[:, 0].mean())
                    print(f"machine load(greedy): ", save_result[:, 1].mean())
                    print(f"time: ", save_result[:, 2].mean())
                    print(f"min machine load(greedy): ", save_result[:, 1].min())
                else:
                    print("Test mode: Sample")
                    save_result = test_sampling_strategy(data[0], model[0], config.sample_times, config.seed_test)
                    print("testing results:")
                    print(f"makespan(sampling): ", save_result[:, 0].mean())
                    print(f"machine load(sampling): ", save_result[:, 1].mean())
                    print(f"time: ", save_result[:, 2].mean())
                    print(f"min machine load(sampling): ", save_result[:, 1].min())
                np.save(save_path, save_result)

if __name__ == '__main__':
    main(configs, False)
    # main(configs, True)





# def test_greedy_strategy(data_set, model_path, seed):
#     """
#         test the model on the given data using the greedy strategy
#     :param data_set: test data
#     :param model_path: the path of the model file
#     :param seed: the seed for testing
#     :return: the test results including the makespan and time
#     """

#     test_result_list = []

#     setup_seed(seed)
#     ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
#     ppo.policy.eval()

#     n_j = data_set[0][0].shape[0]
#     n_op, n_m = data_set[1][0].shape
#     env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)

#     for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):

#         state = env.set_initial_data([data_set[0][i]], [data_set[1][i]])
#         t1 = time.time()
#         while True:

#             with torch.no_grad():
#                 pi, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [1, N, 8]
#                                    op_mask=state.op_mask_tensor,  # [1, N, N]
#                                    candidate=state.candidate_tensor,  # [1, J]
#                                    fea_m=state.fea_m_tensor,  # [1, M, 6]
#                                    mch_mask=state.mch_mask_tensor,  # [1, M, M]
#                                    comp_idx=state.comp_idx_tensor,  # [1, M, M, J]
#                                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
#                                    fea_pairs=state.fea_pairs_tensor)  # [1, J, M]

#             action = greedy_select_action(pi)
#             state, reward, done = env.step(actions=action.cpu().numpy())
#             if done:
#                 break
#         t2 = time.time()

#         test_result_list.append([env.current_makespan[0], t2 - t1])

#     return np.array(test_result_list)


# def test_sampling_strategy(data_set, model_path, sample_times, seed):
#     """
#         test the model on the given data using the sampling strategy
#     :param data_set: test data
#     :param model_path: the path of the model file
#     :param seed: the seed for testing
#     :return: the test results including the makespan and time
#     """
#     setup_seed(seed)
#     test_result_list = []
#     ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
#     ppo.policy.eval()

#     n_j = data_set[0][0].shape[0]
#     n_op, n_m = data_set[1][0].shape
#     from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
#     env = FJSPEnvForSameOpNums(n_j, n_m)

#     for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
#         # copy the testing environment
#         JobLength_dataset = np.tile(np.expand_dims(data_set[0][i], axis=0), (sample_times, 1))
#         OpPT_dataset = np.tile(np.expand_dims(data_set[1][i], axis=0), (sample_times, 1, 1))

#         state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
#         t1 = time.time()
#         while True:

#             with torch.no_grad():
#                 pi, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [100, N, 8]
#                                    op_mask=state.op_mask_tensor,  # [100, N, N]
#                                    candidate=state.candidate_tensor,  # [100, J]
#                                    fea_m=state.fea_m_tensor,  # [100, M, 6]
#                                    mch_mask=state.mch_mask_tensor,  # [100, M, M]
#                                    comp_idx=state.comp_idx_tensor,  # [100, M, M, J]
#                                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [100, J, M]
#                                    fea_pairs=state.fea_pairs_tensor)  # [100, J, M]

#             action_envs, _ = sample_action(pi)
#             state, _, done = env.step(action_envs.cpu().numpy())
#             if done.all():
#                 break

#         t2 = time.time()
#         best_makespan = np.min(env.current_makespan)
#         test_result_list.append([best_makespan, t2 - t1])

#     return np.array(test_result_list)


# def main(config, flag_sample):
#     """
#         test the trained model following the config and save the results
#     :param flag_sample: whether using the sampling strategy
#     """
#     setup_seed(config.seed_test)
#     if not os.path.exists('./test_results'):
#         os.makedirs('./test_results')

#     # collect the path of test models
#     test_model = []

#     for model_name in config.test_model:
#         test_model.append((f'./trained_network/{config.model_source}/{model_name}.pth', model_name))

#     # collect the test data
#     test_data = pack_data_from_config(config.data_source, config.test_data)

#     if flag_sample:
#         model_prefix = "DANIELS"
#     else:
#         model_prefix = "DANIELG"

#     for data in test_data:
#         print("-" * 25 + "Test Learned Model" + "-" * 25)
#         print(f"test data name: {data[1]}")
#         print(f"test mode: {model_prefix}")
#         save_direc = f'./test_results/{config.data_source}/{data[1]}'
#         if not os.path.exists(save_direc):
#             os.makedirs(save_direc)

#         for model in test_model:
#             save_path = save_direc + f'/Result_{model_prefix}+{model[1]}_{data[1]}.npy'
#             if (not os.path.exists(save_path)) or config.cover_flag:
#                 print(f"Model name : {model[1]}")
#                 print(f"data name: ./data/{config.data_source}/{data[1]}")

#                 if not flag_sample:
#                     print("Test mode: Greedy")
#                     result_5_times = []
#                     # Greedy mode, test 5 times, record average time.
#                     for j in range(5):
#                         result = test_greedy_strategy(data[0], model[0], config.seed_test)
#                         result_5_times.append(result)
#                     result_5_times = np.array(result_5_times)

#                     save_result = np.mean(result_5_times, axis=0)
#                     print("testing results:")
#                     print(f"makespan(greedy): ", save_result[:, 0].mean())
#                     print(f"time: ", save_result[:, 1].mean())

#                 else:
#                     # Sample mode, test once.
#                     print("Test mode: Sample")
#                     save_result = test_sampling_strategy(data[0], model[0], config.sample_times, config.seed_test)
#                     print("testing results:")
#                     print(f"makespan(sampling): ", save_result[:, 0].mean())
#                     print(f"time: ", save_result[:, 1].mean())
#                 np.save(save_path, save_result)


# if __name__ == '__main__':
#     main(configs, False)
#     # main(configs, True)
