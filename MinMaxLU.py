import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load TensorBoard events using TFRecordDataset
# logdir = 'logs/training/NSFNet+GEANT2-uniform-sp/batch25-lr0.0003-epsilon0.1-gae0.9-clip0.2-gamma0.95-period50-epoch3/size16-iters8-min_max-nnsize64-drop0.15-tanh'
logdir ='logs/training/NSFNet+GEANT2-gravity_1-sp/batch25-lr0.0003-epsilon0.1-gae0.9-clip0.2-gamma0.95-period50-epoch3/size16-iters8-min_max-nnsize64-drop0.15-tanh'
file_pattern = logdir + '/events*'
dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
print("dataset",dataset)
link_utilization_values_GEANT2 = []
link_utilization_values_NSFNet = []
link_utilization_values_GBN = []

# reward_GEANT2=[]
# reward_NSFNet=[]
# reward_GBN=[]

# Extract link utilization values

    # print("raw_record.numpy",raw_record.numpy)
    # for value in event.summary.value:
        # 定义字典用于存储累积和和计数

reward_accumulators = {}
reward_lists = {}

for raw_record in dataset:
    event = tf.compat.v1.Event.FromString(raw_record.numpy())
    for value in event.summary.value:
        if 'Eval' in value.tag and 'Reward' in value.tag:
            env, _, _, reward_number = value.tag.split(' ')
            reward_number = int(reward_number)

            # 获取 Reward 的值
            tensor_proto = value.tensor
            tensor_content = tensor_proto.tensor_content
            value_float = tf.make_ndarray(tensor_proto).item()
            reward_value = value_float

            # 初始化累积和和计数
            if env not in reward_accumulators:
                reward_accumulators[env] = []
                reward_lists[env] = []

            # 确保 reward_number 对应的列表存在
            while len(reward_accumulators[env]) <= reward_number:
                reward_accumulators[env].append(0.0)
                reward_lists[env].append([])

            # 累积和和计数
            reward_accumulators[env][reward_number] += reward_value
            reward_lists[env][reward_number].append(reward_value)

# 计算均值并组织成列表
reward_results = {}
for env, rewards in reward_accumulators.items():
    reward_results[env] = [sum_list / len(reward_list) if len(reward_list) > 0 else 0.0 for sum_list, reward_list in zip(rewards, reward_lists[env])]

# 输出结果
max_num_rewards = max([len(rewards) for rewards in reward_results.values()])

for env, rewards in reward_results.items():
    print(f"rewards_{env} =", rewards)

time_steps = list(range(1, max_num_rewards + 1))
plt.figure(figsize=(10, 6))
for env, rewards in reward_results.items():
    plt.plot(time_steps, rewards, marker='o', label=env)
plt.xlabel('Episode Length (T)')
plt.ylabel('Max Link Utilization (LU_max_values)')
plt.title('Max Link Utilization (LU_max_values) Evolution over Time')
plt.legend()
plt.grid(True)
plt.show()