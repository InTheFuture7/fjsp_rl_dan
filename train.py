from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import load_data_from_files, CaseGenerator, SD2_instance_generator
from common_utils import strToSuffix, setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from copy import deepcopy
import os
import random
import time
import sys
from model.PPO import PPO_initialize
from model.PPO import Memory

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)

class Trainer:
    def __init__(self, config):

        self.n_j = config.n_j
        self.n_m = config.n_m  # 机器数
        self.low = config.low
        self.high = config.high
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config.data_source
        self.config = config
        self.max_updates = config.max_updates  # 每种环境的更新次数 default=1000
        self.reset_env_timestep = config.reset_env_timestep  # 重置环境的时间间隔 default=20
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs

        # 训练结果保存路径
        if not os.path.exists(f'./trained_network/{self.data_source}'):
            os.makedirs(f'./trained_network/{self.data_source}')
        if not os.path.exists(f'./train_log/{self.data_source}'):
            os.makedirs(f'./train_log/{self.data_source}')

        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        # todo:这部分是限定训练集为 SD1 或 SD2 吗？
        if self.data_source == 'SD1':
            self.data_name = f'{self.n_j}x{self.n_m}'
        elif self.data_source == 'SD2':
            self.data_name = f'{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}'

        self.vali_data_path = f'./data/data_train_vali/{self.data_source}/{self.data_name}'
        self.test_data_path = f'./data/{self.data_source}/{self.data_name}'
        self.model_name = f'{self.data_name}{strToSuffix(config.model_suffix)}'

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
        self.test_data = load_data_from_files(self.test_data_path)
        # validation data set
        vali_data = load_data_from_files(self.vali_data_path)

        if self.data_source == 'SD1':
            # 如果作业操作数不同，应使用 FJSPEnvForVariousOpNums
            self.vali_env = FJSPEnvForVariousOpNums(self.n_j, self.n_m)
        elif self.data_source == 'SD2':
            # 如果所有作业操作数相同，使用 FJSPEnvForSameOpNums
            self.vali_env = FJSPEnvForSameOpNums(self.n_j, self.n_m)

        self.vali_env.set_initial_data(vali_data[0], vali_data[1])

        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                # 生成 20 组工件信息，20组工件工序的机器加工时间
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                #
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                # 重设环境状态
                state = self.env.reset()
            # todo 初始化奖励
            ep_rewards = - deepcopy(self.env.init_quality)

            while True:
                # state store
                self.memory.push(state)
                with torch.no_grad():
                    pi_envs, vals_envs = self.ppo.policy_old(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                                             op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor,  # [sz_b, J]
                                                             fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                                             mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                                             comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                                             fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]
                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # # state transition
                # # todo: 这里传入 action 就携带了 env 的信息，包括 env.current_makespan
                # # 更新完工时间
                # state, reward, done = self.env.step(actions=action_envs.cpu().numpy())

                # ep_rewards += reward
                # reward = torch.from_numpy(reward).to(device)

                # ================ alpha、beta 参数 ============================
                print(type(self.env))  # 添加这行代码以确认self.env的类型
                # 下面代码报错ValueError: not enough values to unpack (expected 4, got 3)
                # todo: 为什么下面计算 reward 时，直接将makespan的reward和机器总负载加权？为什么不是先求出machine_total_load的reward？？
                # 状态转换
                state, makespan_reward, machine_total_load, done = self.env.step(actions=action_envs.cpu().numpy())

                # 计算新的奖励
                alpha = 0.7  # 权重系数，可以根据需要调整
                # beta = 1e-3  # 机器总负载的缩放因子，用于平衡两个目标的量级
                # reward = alpha * makespan_reward - (1 - alpha) * beta * machine_total_load

                reward = alpha * makespan_reward - (1 - alpha) * machine_total_load

                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)

                # ============================================

                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                if self.data_source == "SD1":
                    vali_result = self.validate_envs_with_various_op_nums().mean()
                else:
                    vali_result = self.validate_envs_with_same_op_nums().mean()

                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                self.save_validation_log()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            ep_et = time.time()
            
            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))

        self.train_et = time.time()

        # log results
        self.save_training_log()

    def save_training_log(self):
        """
            save reward data & validation makespan data (during training) and the entire training time
        """
        file_writing_obj = open(f'./train_log/{self.data_source}/' + 'reward_' + self.model_name + '.txt', 'w')
        file_writing_obj.write(str(self.log))

        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))

        file_writing_obj3 = open(f'./train_time.txt', 'a')
        file_writing_obj3.write(
            f'model path: ./DANIEL_FJSP/trained_network/{self.data_source}/{self.model_name}\t\ttraining time: '
            f'{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n')

    def save_validation_log(self):
        """
            save the results of validation
        """
        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))

    def sample_training_instances(self):
        """
            sample training instances following the config,
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl"
        :return: new training instances
        dataset_JobLength：每个工件的工序数量
        dataset_OpPT：从上往下依次是第1个工件的第1个工序在5台机器上的加工时间，第1个工件的第2个工序……第1个工件的第5个工序……，第2个工件的第1个工序
        """
        # Generate the job length list
        # 如果参考数据集是 SD1，那么基于 prepare_JobLength 来生成数据（可变工序数）；否则，直接根据 self.confjg 生成数据
        prepare_JobLength = [random.randint(self.op_per_job_min, self.op_per_job_max) for _ in range(self.n_j)]
        # Print the generated job length list
        # print(f"Generated Job Lengths: {prepare_JobLength}")

        dataset_JobLength = []
        dataset_OpPT = []

        for i in range(self.num_envs):
            # print(f"Generating sample instance {i + 1}/{self.num_envs}")

            if self.data_source == 'SD1':
                case = CaseGenerator(self.n_j, self.n_m, self.op_per_job_min, self.op_per_job_max,
                                     nums_ope=prepare_JobLength, path='./test', flag_doc=False)
                JobLength, OpPT, _ = case.get_case(i)
            else:
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config)

            # Print the details of the generated instance
            # print(f"Instance {i + 1}: Job Lengths: {JobLength}")
            # print(f"Instance {i + 1}: Operation Processing Times (OpPT): {OpPT}")

            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)

        # Print the final datasets
        # print("Final Dataset Job Lengths: ", dataset_JobLength)
        # print("Final Dataset OpPT: ", dataset_OpPT)

        # dataset_JobLength：每个工件的工序数
        # dataset_OpPT：从上往下依次是第1个工件的第1个工序在5台机器上的加工时间，第1个工件的第2个工序……第1个工件的第5个工序……，第2个工件的第1个工序
        return dataset_JobLength, dataset_OpPT

    def validate_envs_with_same_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor,
                                        candidate=state.candidate_tensor,  # [sz_b, J]
                                        fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def validate_envs_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor[batch_idx],
                                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor[batch_idx])  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def save_model(self):
        """
            save the model
        """
        torch.save(self.ppo.policy.state_dict(), f'./trained_network/{self.data_source}'
                                                 f'/{self.model_name}.pth')

    def load_model(self):
        """
            load the trained model
        """
        model_path = f'./trained_network/{self.data_source}/{self.model_name}.pth'
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))


def main():
    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()