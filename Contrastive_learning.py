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
parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--num_epoch', default=300, type=int)
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
parser.add_argument('--PATH_Checkpoint', default="./checkpoint/", type=str)
parser.add_argument('--lambda_1', default=1, type=float)
parser.add_argument('--lambda_2', default=0.5, type=float)
parser.add_argument('--lambda_3', default=0.4, type=float)
args = parser.parse_args()

# Training
def train(epoch,trainloader):
    print('\nTraining')
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, ID, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        SL_weights = SelectiveLearning_Uniform(targets)
        feature_encode, ID_feature_encode,label_encodes, pred = net(x=inputs, IDs=ID)
        loss_bce = F.binary_cross_entropy_with_logits(pred,targets,reduction='mean',weight=SL_weights).to(device)
        loss_contra = Contra_loss(feature_encode, label_encodes, targets).to(device)
        loss_orthogonal = O_loss(feature_encode, ID_feature_encode)
        loss = args.lambda_1 * loss_bce + args.lambda_2 * loss_contra + args.lambda_3 * loss_orthogonal
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    loss_avg = train_loss / (batch_idx+1)
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

def main():

    F1 = []
    loss_dict = {}
    acc_dict = {}
    F1_dict = {}
    start_epoch = -1

    # Initialize early stopping parameters
    best_F1 = float('-inf')  # Track the best F1 score
    patience = 5  # How many epochs to wait before stopping
    epochs_without_improvement = 0  # Counter for early stopping

    for epoch in range(start_epoch + 1, num_epoch):
        print("Fold_{}".format(args.fold) + "\n")
        tra_loss = train(epoch, trainloader)
        val_loss, avg_acc, avg_F1, aus_acc, aus_f1 = val(epoch, valloader)
        scheduler.step(avg_F1)
        F1.append(avg_F1)
        acc_dict.update({'Avg_acc': avg_acc})
        F1_dict.update({'Avg_F1':avg_F1})
        loss_dict.update({'train_loss': tra_loss})
        loss_dict.update({'val_loss': val_loss})
        writer1 = SummaryWriter(args.Descriptions)
        writer1.add_scalars('loss_fold{}'.format(args.fold), loss_dict, global_step=epoch)
        writer1.add_scalars('AUs_acc_fold{}'.format(args.fold), aus_acc, global_step=epoch)
        writer1.add_scalars('AUs_F1_fold{}'.format(args.fold), aus_f1, global_step=epoch)
        writer1.add_scalars('Accuracy_fold{}'.format(args.fold),acc_dict,global_step=epoch)
        writer1.add_scalars('F1_{}'.format(args.fold), F1_dict, global_step=epoch)
        # Early stopping logic
        if avg_F1 > best_F1:
            best_F1 = avg_F1
            epochs_without_improvement = 0  # Reset the counter

            # Report the best F1 score
            print(f"New best F1 score: {best_F1:.4f}")

            # Save the best model's state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'init_of_embed': net.init_of_embed
            }
            torch.save(checkpoint, args.PATH_Checkpoint + args.Descriptions + "_best.pth")


        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement. Best F1 score: {best_F1:.4f}")
                break

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
    #check CUDA
    if torch.cuda.is_available():
        device= args.device
    else:
        device='cpu'

    if args.dataset == "BP4D":
        trainset = BP4Ddataset(args.BP4D_PATH_dataset, args.PATH_dataFile, args.BP4D_PATH_labels, mode="train",
                               fold=args.fold, ID_required=True)
        trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True)

        valset = BP4Ddataset(args.BP4D_PATH_dataset, args.PATH_dataFile, args.BP4D_PATH_labels, mode="val",
                             fold=args.fold, ID_required=True)
        valloader = DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=False)
    else:

        trainset = DISFAdataset(args.DISFA_PATH_dataset, args.PATH_dataFile, args.DISFA_PATH_labels,
                                mode="train",
                                fold=args.fold, ID_required=True)
        trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True)

        valset = DISFAdataset(args.DISFA_PATH_dataset, args.PATH_dataFile, args.DISFA_PATH_labels,
                              mode="val",
                              fold=args.fold, ID_required=True)
        valloader = DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=False)

    Training_IDs = trainset.IDs

    #Models
    net = RepresentationModel(args.fold, args.PATH_backbone, Training_IDs, au_keys, args.dataset)

    optimizer = optim.SGD(net.parameters(), lr=train_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-3,threshold_mode='abs')
    Contra_loss = ContrastiveLoss()
    O_loss = OrthogonalLoss()
    main()

