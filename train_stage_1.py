import numpy as np
from torch.utils.data import  DataLoader
import torch.optim as optim
from tqdm import tqdm
from datasets import *
import torch.nn.functional as F
import warnings
import argparse
from torch.utils.tensorboard import SummaryWriter
from representation_model import IDNet
from loss_func import IDLoss
from utils import SelectiveLearning_Uniform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# args and settings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--num_epoch', default=200, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--Descriptions', default="Test", type=str)
parser.add_argument('--PATH_dataFile', default="./dataSplitFile/", type=str)
parser.add_argument('--fold', default=1, type=int)
parser.add_argument('--dataset', default="BP4D", type=str, help="choose from DISFA, BP4D")
parser.add_argument('--BP4D_PATH_dataset', default="/home/cmp3liant/tangzhenglian/", type=str)
parser.add_argument('--BP4D_PATH_labels', default="/home/cmp3liant/tangzhenglian/AU_OCC_filtered/", type=str)
parser.add_argument('--DISFA_PATH_dataset', default="/home/cmp3liant/tangzhenglian/", type=str)
parser.add_argument('--DISFA_PATH_labels', default="/home/cmp3liant/tangzhenglian/DISFA_Labels/", type=str)
parser.add_argument('--PATH_backbone', default="./checkpoint/", type=str)
parser.add_argument('--lambda_', default=0.4, type=float)
args = parser.parse_args()

# Training
def train(epoch):
    print('\nTraining')
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, ID, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        label_encodes, pred,_ = net(x=inputs,IDs=ID)
        SL_weights = SelectiveLearning_Uniform(targets)
        loss_bce = F.binary_cross_entropy_with_logits(pred,targets,reduction='mean',weight=SL_weights).to(device)
        loss_id = IDLoss(label_encodes)
        loss = args.lambda_ * loss_bce + (1 - args.lambda_) * loss_id
        train_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
    loss_avg = train_loss / (batch_idx+1)
    print("\nAvg_Loss:")
    print(loss_avg)
    return loss_avg, label_encodes


def visualize_label_encodes_pca(label_encodes, epoch, fold):
    # Ensure folders exist
    folder = f"./plots_{fold}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    if epoch % 5 != 0:
        return

    pca = PCA(n_components=2)
    pca_results = {
        "ID": pca.fit_transform(label_encodes.reshape(-1, label_encodes.shape[2]).cpu().detach().numpy()),
        "AU": pca.fit_transform(
            label_encodes.transpose(0, 1).reshape(-1, label_encodes.shape[2]).cpu().detach().numpy())
    }

    IDN = np.arange(0, label_encodes.shape[1] / 2, 0.5, dtype=int)
    for i in range(1, label_encodes.shape[0]):
        IDN = np.hstack((IDN, np.arange(0 + i, label_encodes.shape[1] / 2 + i, 0.5, dtype=int)))

    AUN = np.repeat(np.arange(label_encodes.shape[1]), label_encodes.shape[0])

    df = pd.DataFrame({
        "pca-ID-one": pca_results["ID"][:, 0],
        "pca-ID-two": pca_results["ID"][:, 1],
        "pca-AU-one": pca_results["AU"][:, 0],
        "pca-AU-two": pca_results["AU"][:, 1],
        "y_ID": IDN,
        "y_AU": AUN
    })

    # Plot PCA results
    for label_type, n_colors in [("ID", label_encodes.shape[0]), ("AU", label_encodes.shape[1])]:
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=f"pca-{label_type}-one",
            y=f"pca-{label_type}-two",
            hue=f"y_{label_type}",
            palette=sns.color_palette("husl", n_colors=n_colors),
            data=df,
            legend="full",
            alpha=0.9
        )
        plt.savefig(f"{folder}/PCA_{label_type}_{epoch}.png")

def main():
    loss_dict = {}
    start_epoch = -1
    for epoch in range(start_epoch + 1, num_epoch):
        print("Fold_{}".format(args.fold) + "\n")
        tra_loss, label_encodes = train(epoch)
        visualize_label_encodes_pca(label_encodes, epoch, args.fold)
        loss_dict.update({'train_loss': tra_loss})
        writer1 = SummaryWriter(args.Descriptions)
        writer1.add_scalars('loss_fold{}'.format(args.fold), loss_dict, global_step=epoch)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, args.PATH_backbone + args.Descriptions + ".pth")

# ---------------------------------------------------------------------------------
if __name__=="__main__":
    # parameters
    batchsize = args.batchsize
    train_lr = args.lr
    num_epoch = args.num_epoch
    if args.dataset == "BP4D":
        au_keys = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
    else:
        au_keys = ['AU01', 'AU02', 'AU04', 'AU06', 'AU09', 'AU12', 'AU25', 'AU26']
    if torch.cuda.is_available():
        device= args.device
    else:
        device='cpu'

    #Models
    if args.dataset == "BP4D":
        trainset = BP4Ddataset(args.BP4D_PATH_dataset, args.PATH_dataFile, args.BP4D_PATH_labels, mode="train",
                               fold=args.fold, ID_required=True)
        trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True)
    else:

        trainset = DISFAdataset(args.DISFA_PATH_dataset, args.PATH_dataFile, args.DISFA_PATH_labels,
                                mode="train",
                                fold=args.fold, ID_required=True)
        trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True)

    Training_IDs = trainset.IDs

    net = IDNet(args.fold, args.PATH_backbone, Training_IDs, au_keys, args.dataset)

    optimizer = optim.SGD(net.parameters(), lr=train_lr, momentum=0.9, weight_decay=5e-4)

    IDLoss = IDLoss()

    main()

