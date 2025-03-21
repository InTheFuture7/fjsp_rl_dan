本项目基于 https://github.com/wrqccc/FJSP-DRL 做修改，计划改为多目标调度，目前仍在修改中...


# FJSP-DRL

This repository is the official implementation of the paper “[Flexible Job Shop Scheduling via Dual Attention Network Based Reinforcement Learning](https://doi.org/10.1109/TNNLS.2023.3306421)”. *IEEE Transactions on Neural Networks and Learning Systems*, 2023.

## Quick Start

### requirements

- python $=$ 3.7.11
- argparse $=$ 1.4.0
- numpy $=$ 1.21.6
- ortools $=$ 9.3.10497
- pandas $=$ 1.3.5
- torch $=$ 1.11.0+cu113
- torchaudio $=$ 0.11.0+cu113
- torchvision $=$ 0.12.0+cu113
- tqdm $=$ 4.64.0

### introduction

- `data` saves the instance files including testing instances (in the subfolder `BenchData`, `SD1` and `SD2`) and validation instances (in the subfolder `data_train_vali`) .
- `model` contains the implementation of the proposed framework.
- `or_solution` saves the results solved by Google OR-Tools.
- `test_results`saves the results solved by priority dispatching rules(优先调度规则PDR) and DRL models.
- `train_log` saves the training log of models, including information of the reward and validation makespan.
- `trained_network` saves the trained models.
- `common_utils.py` contains some useful functions (including the implementation of priority dispatching rules mentioned in the paper 优先调度规则的实现) .
- `data_utils.py` is used for data generation, reading and format conversion.
- `fjsp_env_same_op_nums.py` and `fjsp_env_various_op_nums.py` are implementations of fjsp environments, describing fjsp instances with the same number of operations and different number of operations, respectively.
- `ortools_solver.py` is used for solving the instances by Google OR-Tools.
- `params.py` defines parameters settings.
- `print_test_result.py` is used for printing the experimental results into an Excel file.
- `test_heuristic.py` is used for solving the instances by priority dispatching rules.
- `test_trained_model.py` is used for evaluating the models.
- `train.py` is used for training.

### train

基于所有代码做分析，如果依次运行以下两个命令，是否能够获得，针对Brandimarte 数据集在两个目标函数（最小最大完工时间和最小机器总负载）下的最优车间调度方案？

```python
python train.py # train the model on 10x5 FJSP instances using SD2

# options (Validation instances of corresponding size should be prepared in ./data/data_train_vali/{data_source})
python train.py --n_j 10		# number of jobs for training/validation instances
			    --n_m 5			# number of machines for training/validation instances
    			--data_source SD2	# data source (SD1 / SD2)
        		--data_suffix mix	# mode for SD2 data generation
            					# 'mix' is the default mode as defined in the paper  #todo 理解 mix 参数
                				# 'nf' means 'no flexibility' (generating JSP data) 
        		--model_suffix demo	# annotations for the model  #todo 理解参数

python train.py --n_j 10  --n_m 5  --data_source SD2  --data_suffix mix  --model_suffix demo
```

### evaluate

```python
python test_trained_model.py # evaluate the model trained on '10x5+mix' of SD2 using the testing instances of the same size using the greedy strategy

# options (Model files should be prepared in ./trained_network/{model_source})
python test_trained_model.py    --data_source SD2	# source of testing instances
				                --model_source SD2	# source of instances that the model trained on
    				            --test_data 10x5+mix	# list of instance names for testing
        			            --test_model 10x5+mix	# list of model names for testing
            			        --test_mode False	# whether using the sampling strategy，default is False(greedy strategy)
                		        --sample_times 100	# set the number of sampling times 为每个测试实例生成 100 个解决方案，取最优的

# 使用训练好的模型 ./trained_network/{model_source}/{test_model} 用于测试数据集 {data_source}/{test_data}
python test_trained_model.py  --data_source BenchData  --test_data test  --model_source SD2  --test_model 10x5+mix+demo  --test_mode True  --sample_times 100
```

## Cite the paper

```
@ARTICLE{10246328,
  author={Wang, Runqing and Wang, Gang and Sun, Jian and Deng, Fang and Chen, Jie},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TNNLS.2023.3306421}
}
```

## Reference

- https://github.com/songwenas12/fjsp-drl/
- https://github.com/zcaicaros/L2D
- https://github.com/google/or-tools
- https://github.com/Diego999/pyGAT
