import os
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

# Define the path and the Action Units (AUs)
path = "/vol/lian/datasets/AU_OCC/"
list_csv = glob(os.path.join(path + '*.csv'))

au_keys = ['AU01','AU02','AU04','AU06','AU07','AU10','AU12','AU14','AU15','AU17','AU23','AU24']
au_indices = [1,2,4,6,7,10,12,14,15,17,23,24] # Corresponding indices for AUs in csv

# Read all csv files, extract AUs and concatenate all frames
all_frames = []
for csv_file in list_csv:
    csv = pd.read_csv(csv_file)
    AUs = csv.iloc[:, au_indices].values # Select the AU columns directly
    all_frames.extend(AUs)
all_frames = np.array(all_frames)

# Initialize arrays for storing results
au_len = len(au_keys)
co_occ_num = np.zeros((au_len, au_len))
co_neg_num = np.zeros((au_len, au_len))

# Calculate co-occurrence and co-absence for each pair of AUs
for i in range(au_len):
    for j in range(au_len):
        if i != j:
            co_occ_num[i, j] = np.sum((all_frames[:, i]==1) & (all_frames[:, j]==1))
            co_neg_num[i, j] = np.sum((all_frames[:, i]==0) & (all_frames[:, j]==0))

# Initialize arrays for storing probabilities
co_occ_pro = np.zeros((au_len, au_len))
co_occ_con_pro = np.zeros((au_len, au_len))

# Calculate occurrence number for each AU
occ_num_arr = np.sum(all_frames==1, axis=0)

# Calculate conditional co-occurrence probability and co-occurrence probability
for i in range(au_len):
    for j in range(au_len):
        if i != j:
            co_occ_pro[i, j] = co_occ_num[i, j] / len(all_frames)
            co_occ_con_pro[i, j] = 0.5 * (co_occ_num[i, j] / occ_num_arr[i] + co_neg_num[i, j] / (len(all_frames) - occ_num_arr[i]))

# Calculate co-occurrence coefficients
co_occ_coe = np.where(co_occ_con_pro > 0.5, (co_occ_con_pro - 0.5) * 2, 0)

# Adjust conditional co-occurrence probabilities for visualization
co_occ_con_pro = np.abs((co_occ_con_pro - 0.5) * 2)
co_occ_con_pro = np.round(co_occ_con_pro, 1)

# Generate a heatmap of the adjusted conditional co-occurrence probabilities
plt.subplots(figsize=(20,15))
sns.set(font_scale=4)
ax = sns.heatmap(co_occ_con_pro, xticklabels=au_keys, yticklabels=au_keys, square=True)
ax.set_xticklabels(ax.get_yticklabels(), rotation=90, fontsize=45)
ax.set_yticklabels(ax.get_xticklabels(), rotation=360, fontsize=45)
plt.show()
