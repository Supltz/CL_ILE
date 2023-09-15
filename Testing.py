from torch.utils.data import  DataLoader
import torch.optim as optim
from tqdm import tqdm
from datasets import *
import torch.nn.functional as F
import warnings
import argparse
from torch.utils.tensorboard import SummaryWriter
from representation_model import RepresentationModel
from loss_func import ContrastiveLoss,OrthogonalLoss
import torch
from utils import SelectiveLearning_Uniform, Get_ALL, compute_metrics


# args and settings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--PATH_dataFile', default="./dataSplitFile/", type=str)
parser.add_argument('--dataset', default="BP4D", type=str, help="choose from DISFA, BP4D")
parser.add_argument('--BP4D_PATH_dataset', default="/home/cmp3liant/tangzhenglian/", type=str)
parser.add_argument('--BP4D_PATH_labels', default="/home/cmp3liant/tangzhenglian/AU_OCC_filtered/", type=str)
parser.add_argument('--DISFA_PATH_dataset', default="/home/cmp3liant/tangzhenglian/", type=str)
parser.add_argument('--DISFA_PATH_labels', default="/home/cmp3liant/tangzhenglian/DISFA_Labels/", type=str)
parser.add_argument('--PATH_backbone', default="./checkpoint/", type=str)
args = parser.parse_args()

@torch.no_grad()
def val(valloader):
    print('\nValidation:')
    net.eval()
    val_loss = 0
    TPs_in_valset = [0 for _ in range(len(au_keys))]
    TNs_in_valset = [0 for _ in range(len(au_keys))]
    FNs_in_valset = [0 for _ in range(len(au_keys))]
    FPs_in_valset = [0 for _ in range(len(au_keys))]

    for batch_idx, (inputs, ID, targets) in enumerate(tqdm(valloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        _,label_embeds,pred = net(x=inputs, IDs=ID, val_mode=True)
        loss = F.binary_cross_entropy_with_logits(pred, targets, reduction='mean').to(device)
        val_loss += loss.item()
        AU_TP, AU_TN, AU_FN, AU_FP = Get_ALL(pred, targets)
        TPs_in_valset = [a + b for a, b in zip(TPs_in_valset, AU_TP)]
        TNs_in_valset = [a + b for a, b in zip(TNs_in_valset, AU_TN)]
        FNs_in_valset = [a + b for a, b in zip(FNs_in_valset, AU_FN)]
        FPs_in_valset = [a + b for a, b in zip(FPs_in_valset, AU_FP)]

    precision, recall, F1, acc = compute_metrics(TPs_in_valset, TNs_in_valset, FNs_in_valset, FPs_in_valset)

    Aus_acc = {k: v for k, v in zip(au_keys, acc)}
    Aus_F1 = {k: v for k, v in zip(au_keys, F1)}
    avg_loss = val_loss / (batch_idx + 1)
    val_acc = sum(acc) / len(acc)
    val_F1 = sum(F1) / len(F1)

    print("Accuracy:", Aus_acc)
    print("F1_Score:", Aus_F1)
    print("Avg_Loss:", avg_loss)
    print("Avg_Acc:", val_acc)
    print("Avg_F1:", val_F1)

    return avg_loss, val_acc, val_F1, Aus_acc, Aus_F1

def load_pretrained_model(model, pretrained_path):
    pretrained = torch.load(pretrained_path)
    model_dict = model.state_dict()
    model.init_of_embed = pretrained['init_of_embed']
    pretrained = {k: v for k, v in pretrained["model_state_dict"].items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    for param in model.parameters():
        param.requires_grad = False

# parameters
device = args.device

if args.dataset == "BP4D":
    PRETRAINED_PATHS = ["/home/cmp3liant/tangzhenglian/code/checkpoint/Release_1_best.pth",
                        "/home/cmp3liant/tangzhenglian/code/checkpoint/Release_2_best.pth",
                        "/home/cmp3liant/tangzhenglian/code/checkpoint/Release_3_best.pth"]
    au_keys = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
else:
    PRETRAINED_PATHS = ["/home/cmp3liant/tangzhenglian/code/checkpoint/Release_D_1_best.pth",
                        "/home/cmp3liant/tangzhenglian/code/checkpoint/Release_D_2_best.pth",
                        "/home/cmp3liant/tangzhenglian/code/checkpoint/Release_D_3_best.pth"]
    au_keys = ['AU01', 'AU02', 'AU04', 'AU06', 'AU09', 'AU12', 'AU25', 'AU26']
# check CUDA
if torch.cuda.is_available():
    device = args.device
else:
    device = 'cpu'




for fold in range(1, 4):

    print("Fold_{}".format(fold) + "\n")

    if args.dataset == "BP4D":

        trainset = BP4Ddataset(args.BP4D_PATH_dataset, args.PATH_dataFile, args.BP4D_PATH_labels, mode="train",
                               fold=fold, ID_required=True)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True)

        valset = BP4Ddataset(args.BP4D_PATH_dataset, args.PATH_dataFile, args.BP4D_PATH_labels, mode="val",
                             fold=fold, ID_required=True)
        valloader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=False)
    else:

        valset = DISFAdataset(args.DISFA_PATH_dataset, args.PATH_dataFile, args.DISFA_PATH_labels,
                              mode="val",
                              fold=fold, ID_required=True)
        valloader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=False)

    Training_IDs = trainset.IDs

    # Models
    net = RepresentationModel(fold, args.PATH_backbone, Training_IDs, au_keys, args.dataset)
    load_pretrained_model(net, PRETRAINED_PATHS[fold - 1])
    net.eval()
    val_loss, avg_acc, avg_F1, aus_acc, aus_f1 = val(valloader)


