import numpy as np
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

reward_GEANT2=[]
reward_NSFNet=[]
reward_GBN=[]

# Extract link utilization values
for raw_record in dataset:
    event = tf.compat.v1.Event.FromString(raw_record.numpy())
    # print("raw_record.numpy",raw_record.numpy)
    for value in event.summary.value:
        print("value.tag",value.tag)
        if value.tag == 'Eval/GEANT2 - MEAN Min Max LU':
            tensor_proto = value.tensor
            tensor_content = tensor_proto.tensor_content
            value_float = tf.make_ndarray(tensor_proto).item()
            link_utilization_values_GEANT2.append(value_float)
            # link_utilization_values_GEANT2.append(value.simple_value)
        if value.tag == 'Eval/NSFNet - MEAN Min Max LU':
            tensor_proto = value.tensor
            tensor_content = tensor_proto.tensor_content
            value_float = tf.make_ndarray(tensor_proto).item()
            link_utilization_values_NSFNet.append(value_float)
            # link_utilization_values_NSFNet.append(value.simple_value)
        if value.tag == 'Eval/GBN - MEAN Min Max LU':
            tensor_proto = value.tensor
            tensor_content = tensor_proto.tensor_content
            value_float = tf.make_ndarray(tensor_proto).item()
            link_utilization_values_GBN.append(value_float)
            # link_utilization_values_GBN.append(value.simple_value)



print("reward_NSFNet",reward_NSFNet,len(reward_NSFNet))

# Sort the values
link_utilization_values_GEANT2.sort()
print("link2",link_utilization_values_GEANT2)
# Calculate the CDF
cdf_values = np.arange(1, len(link_utilization_values_GEANT2) + 1) / len(link_utilization_values_GEANT2)

# Plot the CDF
plt.plot(link_utilization_values_GEANT2, cdf_values, marker='o')
plt.xlabel('Link Utilization')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function of Eval/GEANT2')
plt.grid()
plt.show()

# Sort the values
link_utilization_values_NSFNet.sort()
print("link2",link_utilization_values_NSFNet)
# Calculate the CDF
cdf_values = np.arange(1, len(link_utilization_values_NSFNet) + 1) / len(link_utilization_values_NSFNet)

# Plot the CDF
plt.plot(link_utilization_values_NSFNet, cdf_values, marker='o')
plt.xlabel('Link Utilization')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function of Eval/NSFNet')
plt.grid()
plt.show()

# Sort the values
link_utilization_values_GBN.sort()
print("link2",link_utilization_values_GBN)
# Calculate the CDF
cdf_values = np.arange(1, len(link_utilization_values_GBN) + 1) / len(link_utilization_values_GBN)

# Plot the CDF
plt.plot(link_utilization_values_GBN, cdf_values, marker='o')
plt.xlabel('Link Utilization')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function of Eval/GBN')
plt.grid()
plt.show()

