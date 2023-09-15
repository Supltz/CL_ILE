import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
from vit_pytorch import ViT
from vit_pytorch.max_vit import MaxViT

from CoatNet import coatnet_0, coatnet_1, coatnet_2, coatnet_3, coatnet_4
from NewResnets import resnet18, resnet34, resnet50, resnet101, resnet152
from datasets import BP4Ddataset, DISFAdataset
from swin_transformer import swin_transformer_base
from utils import SelectiveLearning_Uniform, Get_ALL, compute_metrics

# args and settings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--Descriptions', default="Test", type=str)
parser.add_argument('--PATH_dataFile', default="./dataSplitFile/", type=str)
parser.add_argument('--model', default="ResNet18", type=str, help="choose from ResNet18,34,50,101,152")
parser.add_argument('--fold', default=1, type=int)
parser.add_argument('--dataset', default="BP4D", type=str, help="choose from DISFA, BP4D")
parser.add_argument('--BP4D_PATH_dataset', default="/vol/lian/datasets/", type=str)
parser.add_argument('--BP4D_PATH_labels', default="/vol/lian/datasets/AU_OCC_filtered/", type=str)
parser.add_argument('--DISFA_PATH_dataset', default="/vol/lian/datasets/", type=str)
parser.add_argument('--DISFA_PATH_labels', default="//vol/lian/datasets/DISFA_Labels/", type=str)

args = parser.parse_args()


# Training
def train(epoch, trainloader):
    print('\nTraining')
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        SL_weights = SelectiveLearning_Uniform(targets)
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='mean', weight=SL_weights).to(device)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss_avg = train_loss / (batch_idx + 1)
    print("\nAvg_Loss:")
    print(loss_avg)
    return loss_avg

@torch.no_grad()
def val(epoch, valloader):
    print('\nValidation:')
    print('\nEpoch(validation): %d' % epoch)
    net.eval()
    val_loss = 0
    TPs_in_valset = [0 for _ in range(len(au_keys))]
    TNs_in_valset = [0 for _ in range(len(au_keys))]
    FNs_in_valset = [0 for _ in range(len(au_keys))]
    FPs_in_valset = [0 for _ in range(len(au_keys))]

    for batch_idx, (inputs, targets) in enumerate(tqdm(valloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='mean').to(device)
        val_loss += loss.item()
        AU_TP, AU_TN, AU_FN, AU_FP = Get_ALL(outputs, targets)
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


def main():
    if args.dataset == "BP4D":
        trainset = BP4Ddataset(args.BP4D_PATH_dataset, args.PATH_dataFile, args.BP4D_PATH_labels, mode="train",
                               fold=args.fold, ID_required=False)
        trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True)

        valset = BP4Ddataset(args.BP4D_PATH_dataset, args.PATH_dataFile, args.BP4D_PATH_labels, mode="val",
                             fold=args.fold, ID_required=False)
        valloader = DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=False)
    else:

        trainset = DISFAdataset(args.DISFA_PATH_dataset, args.PATH_dataFile, args.DISFA_PATH_labels,
                                mode="train",
                                fold=args.fold, ID_required=False)
        trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True)

        valset = DISFAdataset(args.DISFA_PATH_dataset, args.PATH_dataFile, args.DISFA_PATH_labels,
                              mode="val",
                              fold=args.fold, ID_required=False)
        valloader = DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=False)

    F1 = []
    loss_dict = {}
    acc_dict = {}
    F1_dict = {}
    start_epoch = -1
    for epoch in range(start_epoch + 1, num_epoch):
        print("Fold_{}".format(args.fold) + "\n")
        tra_loss = train(epoch, trainloader)
        val_loss, avg_acc, avg_F1, aus_acc, aus_f1 = val(epoch, valloader)
        F1.append(avg_F1)
        acc_dict.update({'Avg_acc': avg_acc})
        F1_dict.update({'Avg_F1': avg_F1})
        loss_dict.update({'train_loss': tra_loss})
        loss_dict.update({'val_loss': val_loss})
        writer1 = SummaryWriter(args.Descriptions)
        writer1.add_scalars('loss_fold{}'.format(args.fold), loss_dict, global_step=epoch)
        writer1.add_scalars('AUs_acc_fold{}'.format(args.fold), aus_acc, global_step=epoch)
        writer1.add_scalars('AUs_F1_fold{}'.format(args.fold), aus_f1, global_step=epoch)
        writer1.add_scalars('Accuracy_fold{}'.format(args.fold), acc_dict, global_step=epoch)
        writer1.add_scalars('F1_{}'.format(args.fold), F1_dict, global_step=epoch)


if __name__ == "__main__":
    # parameters
    batchsize = args.batchsize
    train_lr = args.lr
    num_epoch = args.num_epoch
    if args.dataset == "BP4D":
        au_keys = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
    else:
        au_keys = ['AU01', 'AU02', 'AU04', 'AU06', 'AU09', 'AU12', 'AU25', 'AU26']

    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'

    # Models
    Nets = {"ResNet18": resnet18(backbone=True, num_classes=len(au_keys)),
            "ResNet34": resnet34(backbone=True, num_classes=len(au_keys)),
            "ResNet50": resnet50(backbone=True, num_classes=len(au_keys)),
            "ResNet101": resnet101(backbone=True, num_classes=len(au_keys)),
            "ResNet152": resnet152(backbone=True, num_classes=len(au_keys)),
            "Iv3": models.inception_v3(pretrained=False, aux_logits=False, num_classes=len(au_keys)),
            "ViT": ViT(image_size=224, patch_size=16, num_classes=len(au_keys), dim=1024, depth=6, heads=16, mlp_dim=2048,
                       dropout=0.1, emb_dropout=0.1),
            "Swin": swin_transformer_base(num_classes=len(au_keys)),
            "CoatNet-0": coatnet_0(len(au_keys)),
            "CoatNet-1": coatnet_1(len(au_keys)),
            "CoatNet-2": coatnet_2(len(au_keys)),
            "CoatNet-3": coatnet_3(len(au_keys)),
            "CoatNet-4": coatnet_4(len(au_keys)),
            # Uncomment if you want to try those models
            # "Efficient-1": EfficientNet.from_pretrained('efficientnet-b1', num_classer=len(au_keys),
            # "Efficient-2": EfficientNet.from_pretrained('efficientnet-b2', num_classer=len(au_keys),
            # "Efficient-3": EfficientNet.from_pretrained('efficientnet-b3', num_classer=len(au_keys),
            # "Efficient-4": EfficientNet.from_pretrained('efficientnet-b4', num_classer=len(au_keys),
            # "Efficient-5": EfficientNet.from_pretrained('efficientnet-b5', num_classer=len(au_keys),
            # "Efficient-6": EfficientNet.from_pretrained('efficientnet-b6', num_classer=len(au_keys),
            # "Efficient-7": EfficientNet.from_pretrained('efficientnet-b7', num_classer=len(au_keys),
            "Max-ViT": MaxViT(
                num_classes=len(au_keys),
                dim_conv_stem=64,
                dim=96,
                dim_head=32,
                depth=(2, 2, 5, 2),
                window_size=7,
                mbconv_expansion_rate=4,
                mbconv_shrinkage_rate=0.25,
                dropout=0.1
            )}

    net = Nets[args.model]

    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=train_lr, momentum=0.9, weight_decay=5e-4)

    main()
