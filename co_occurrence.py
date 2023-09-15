import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_co_occurrences(data, au_keys):
    au_len = len(au_keys)

    co_occ_num = np.zeros((au_len, au_len))
    co_neg_num = np.zeros((au_len, au_len))

    # Calculate co-occurrence and co-absence for each pair of AUs
    for i in range(au_len):
        for j in range(au_len):
            if i != j:
                co_occ_num[i, j] = np.sum((data[:, i] == 1) & (data[:, j] == 1))
                co_neg_num[i, j] = np.sum((data[:, i] == 0) & (data[:, j] == 0))

    return co_occ_num, co_neg_num


def calculate_probabilities(co_occ_num, co_neg_num, data, au_keys):
    au_len = len(au_keys)
    occ_num_arr = np.sum(data == 1, axis=0)

    co_occ_con_pro = np.zeros((au_len, au_len))

    for i in range(au_len):
        for j in range(au_len):
            if i != j:
                co_occ_con_pro[i, j] = 0.5 * (
                            co_occ_num[i, j] / occ_num_arr[i] + co_neg_num[i, j] / (len(data) - occ_num_arr[i]))

    return co_occ_con_pro


def visualize_co_occurrences(co_occ_con_pro, au_keys):
    # Adjust and visualize
    co_occ_con_pro = np.abs((co_occ_con_pro - 0.5) * 2)
    co_occ_con_pro = np.round(co_occ_con_pro, 1)

    plt.subplots(figsize=(20, 15))
    sns.set(font_scale=4)
    ax = sns.heatmap(co_occ_con_pro, xticklabels=au_keys, yticklabels=au_keys, square=True)
    ax.set_xticklabels(ax.get_yticklabels(), rotation=90, fontsize=45)
    ax.set_yticklabels(ax.get_xticklabels(), rotation=360, fontsize=45)
    plt.show()
