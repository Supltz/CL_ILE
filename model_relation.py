import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import BP4Ddataset
from tqdm import tqdm
from torch import nn
from representation_model import RepresentationModel

# Constants

PATH_dataFile = "./dataSplitFile/"
PATH_dataset = "/home/cmp3liant/tangzhenglian/"
PATH_labels = "/home/cmp3liant/tangzhenglian/AU_OCC_filtered/"

dataset = "BP4D"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
AU_KEYS = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
PRETRAINED_PATHS = ["/home/cmp3liant/tangzhenglian/code/checkpoint/Release_1_best.pth",
                    "/home/cmp3liant/tangzhenglian/code/checkpoint/Release_2_best.pth",
                    "/home/cmp3liant/tangzhenglian/code/checkpoint/Release_3_best.pth"]

PATH_backbone = "/home/cmp3liant/tangzhenglian/code/checkpoint/"

sigmoid = nn.Sigmoid()


def load_pretrained_model(model, pretrained_path):
    pretrained = torch.load(pretrained_path)
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in pretrained["model_state_dict"].items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    for param in model.parameters():
        param.requires_grad = False


def get_predictions_for_fold(fold):

    valset = BP4Ddataset(PATH_dataset, PATH_dataFile, PATH_labels, mode="val", fold=fold, ID_required=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    val_IDs = valset.IDs
    model = RepresentationModel(fold, PATH_backbone, val_IDs, AU_KEYS, dataset).to(DEVICE)
    load_pretrained_model(model, PRETRAINED_PATHS[fold-1])
    pred_in_fold = []
    model.eval()

    for inputs, ID, _ in tqdm(valloader):
        inputs = inputs.to(DEVICE)
        pred = model(inputs, ID, val_mode=True)
        predicted = sigmoid(pred[-1]).squeeze()
        predicted = (predicted >= 0.5).float()
        pred_in_fold.append(predicted.cpu().detach().numpy())
    return pred_in_fold


pred_all = []
for fold in range(1, 4):
    pred_all.extend(get_predictions_for_fold(fold))



pred_all = np.vstack(pred_all)
# Initialize arrays for storing results
au_len = len(AU_KEYS)
co_occ_num = np.zeros((au_len, au_len))
co_neg_num = np.zeros((au_len, au_len))

# Calculate co-occurrence and co-absence for each pair of AUs
for i in range(au_len):
    for j in range(au_len):
        if i != j:
            co_occ_num[i, j] = np.sum((pred_all[:, i]==1) & (pred_all[:, j]==1))
            co_neg_num[i, j] = np.sum((pred_all[:, i]==0) & (pred_all[:, j]==0))

# Initialize arrays for storing probabilities
co_occ_pro = np.zeros((au_len, au_len))
co_occ_con_pro = np.zeros((au_len, au_len))

# Calculate occurrence number for each AU
occ_num_arr = np.sum(pred_all==1, axis=0)

# Calculate conditional co-occurrence probability and co-occurrence probability
for i in range(au_len):
    for j in range(au_len):
        if i != j:
            co_occ_pro[i, j] = co_occ_num[i, j] / len(pred_all)
            co_occ_con_pro[i, j] = 0.5 * (co_occ_num[i, j] / occ_num_arr[i] + co_neg_num[i, j] / (len(pred_all) - occ_num_arr[i]))

# Calculate co-occurrence coefficients
co_occ_coe = np.where(co_occ_con_pro > 0.5, (co_occ_con_pro - 0.5) * 2, 0)

# Adjust conditional co-occurrence probabilities for visualization
co_occ_con_pro = np.abs((co_occ_con_pro - 0.5) * 2)
co_occ_con_pro = np.round(co_occ_con_pro, 1)

# Generate a heatmap of the adjusted conditional co-occurrence probabilities
plt.subplots(figsize=(20,15))
sns.set(font_scale=4)
ax = sns.heatmap(co_occ_con_pro, xticklabels=AU_KEYS, yticklabels=AU_KEYS, square=True)
ax.set_xticklabels(ax.get_yticklabels(), rotation=90, fontsize=45)
ax.set_yticklabels(ax.get_xticklabels(), rotation=360, fontsize=45)
plt.show()
