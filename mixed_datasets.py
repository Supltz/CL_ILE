import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import argparse
import inspect

from datasets import BP4Ddataset, DISFAdataset
from swin_transformer import swin_transformer_base
from NewResnets import resnet18, resnet50, resnet152
from efficientnet_pytorch import EfficientNet
from representation_model import RepresentationModel, IDNet
from loss_func import ContrastiveLoss,OrthogonalLoss,IDLoss

# args and settings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--num_epoch', default=200, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--Descriptions', default="Test", type=str)
parser.add_argument('--PATH_dataFile', default="./dataSplitFile/", type=str)
parser.add_argument('--BP4D_PATH_dataset', default="/home/cmp3liant/tangzhenglian/", type=str)
parser.add_argument('--BP4D_PATH_labels', default="/home/cmp3liant/tangzhenglian/AU_OCC_filtered/", type=str)
parser.add_argument('--DISFA_PATH_dataset', default="/home/cmp3liant/tangzhenglian/", type=str)
parser.add_argument('--DISFA_PATH_labels', default="/home/cmp3liant/tangzhenglian/DISFA_Labels/", type=str)
parser.add_argument('--PATH_backbone', default="./checkpoint/", type=str)
parser.add_argument('--PATH_Checkpoint', default="./checkpoint/", type=str)
parser.add_argument('--model', default="ResNet18", type=str, required=True)
parser.add_argument('--lambda_1', default=1, type=float)
parser.add_argument('--lambda_2', default=0.5, type=float)
parser.add_argument('--lambda_3', default=0.4, type=float)
parser.add_argument('--lambda_', default=0.4, type=float)
args = parser.parse_args()

# Constants
au_keys = ['AU01','AU02','AU04','AU06','AU07', 'AU09', 'AU10','AU12','AU14','AU15','AU17','AU23','AU24','AU25', 'AU26']

# Device selection
device = args.device if torch.cuda.is_available() else 'cpu'

def get_SL_weights(labels):
    SL_weights = []
    for i in range(len(labels[0])):
        AU = labels[:, i]
        valid = AU != -1
        AU = AU[valid]

        if len(AU) == 0:
            weight = torch.zeros_like(labels[:, i], dtype=torch.float)
        else:
            ratio = sum(AU) / len(AU)
            weight = torch.zeros_like(labels[:, i], dtype=torch.float)
            if ratio > 0.5:
                for j, s in enumerate(torch.where(valid)[0]):
                    if j < round(len(AU)/2) and AU[j] == 1:
                        weight[s] = 1
                    elif AU[j] == 0:
                        weight[s] = 0.5 / (1 - ratio)
            elif ratio < 0.5:
                for j, s in enumerate(torch.where(valid)[0]):
                    if AU[j] == 0 and j < round(len(AU)/2):
                        weight[s] = 1
                    elif AU[j] == 1:
                        weight[s] = 0.5 / ratio

        SL_weights.append(weight)

    return torch.stack(SL_weights).transpose(0, 1).to(device)


def train(epoch, trainloader):
    print(f'\nTraining\nEpoch: {epoch}')
    net.train()
    train_loss = 0

    forward_params = inspect.signature(net.forward).parameters

    for batch_idx, (inputs, IDs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if 'IDs' in forward_params and 'val_mode' in forward_params:
            feature_encode, ID_feature_encode, label_encodes, outputs = net(x=inputs, IDs=IDs)
            SL_weights = get_SL_weights(targets)
            mask = (targets != -1)
            targets_masked = torch.where(mask, targets, torch.tensor(0).long().to(device))
            outputs_masked = torch.where(mask, outputs, torch.tensor(0.).to(device))
            targets_masked = targets_masked.to(torch.float32)
            loss_bce = F.binary_cross_entropy_with_logits(outputs_masked, targets_masked, reduction='mean', weight=SL_weights).to(
                device)
            loss_contra = Contra_loss(feature_encode, label_encodes, targets).to(device)
            loss_orthogonal = O_loss(feature_encode, ID_feature_encode)
            loss = args.lambda_1 * loss_bce + args.lambda_2 * loss_contra + args.lambda_3 * loss_orthogonal
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        elif 'IDs' in forward_params:
            label_encodes, outputs, _ = net(x=inputs, IDs=IDs)
            SL_weights = get_SL_weights(targets)
            mask = (targets != -1)
            targets_masked = torch.where(mask, targets, torch.tensor(0).long().to(device))
            outputs_masked = torch.where(mask, outputs, torch.tensor(0.).to(device))
            targets_masked = targets_masked.to(torch.float32)
            loss_bce = F.binary_cross_entropy_with_logits(outputs_masked, targets_masked, reduction='mean', weight=SL_weights).to(
                device)
            loss_id = IDLoss(label_encodes)
            loss = args.lambda_ * loss_bce + (1 - args.lambda_) * loss_id
            train_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

        else:
            outputs = net(inputs)
            SL_weights = get_SL_weights(targets)
            mask = (targets != -1)
            targets_masked = torch.where(mask, targets, torch.tensor(0).long().to(device))
            outputs_masked = torch.where(mask, outputs, torch.tensor(0.).to(device))
            targets_masked = targets_masked.to(torch.float32)
            loss = F.binary_cross_entropy_with_logits(outputs_masked, targets_masked, weight=SL_weights)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()


    avg_loss = train_loss / (batch_idx + 1)
    print(f"\nAvg_Loss: {avg_loss}")

    return avg_loss

def collate_fn(batch):
    all_AUs = au_keys
    AUs_BP4D = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
    AUs_DISFA = ['AU01', 'AU02', 'AU04', 'AU06', 'AU09', 'AU12', 'AU25', 'AU26']

    inputs, ID, targets = zip(*batch)
    inputs = torch.stack(inputs, 0)
    ID = list(ID)

    targets_padded = []
    for target in targets:
        target_padded = torch.full((len(all_AUs),), -1)
        dataset_AUs = AUs_BP4D if len(target) == len(AUs_BP4D) else AUs_DISFA
        for i, AU in enumerate(all_AUs):
            if AU in dataset_AUs:
                index = dataset_AUs.index(AU)
                target_padded[i] = target[index]
        targets_padded.append(target_padded)

    return inputs, ID, torch.stack(targets_padded, 0)


# ---------------------------------------------------------------------------------
if __name__=="__main__":

    datasets = [
                   BP4Ddataset(args.BP4D_PATH_dataset, args.PATH_dataFile, args.BP4D_PATH_labels, mode="val", fold=i,
                               ID_required=True) for i in [1, 2, 3]
               ] + [
                   DISFAdataset(args.DISFA_PATH_dataset, args.PATH_dataFile, args.DISFA_PATH_labels, mode="val", fold=i,
                                ID_required=True) for i in [1, 2, 3]
               ]
    concat_dataset = ConcatDataset(datasets)
    Training_IDs = []
    for dataset in datasets:
        Training_IDs.extend(dataset.IDs)
    trainloader = DataLoader(concat_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    # Model initialization
    model_choices = {
        "ResNet18": resnet18(backbone=True, num_classes=len(au_keys)),
        "ResNet50": resnet50(backbone=True, num_classes=len(au_keys)),
        "ResNet152": resnet152(backbone=True, num_classes=len(au_keys)),
        "Swin": swin_transformer_base(num_classes=len(au_keys)),
        "Contra": RepresentationModel(0, args.PATH_backbone, Training_IDs, au_keys, None),
        "IDnet": IDNet(0, args.PATH_backbone, Training_IDs, au_keys, None),
        "Efficient-2": EfficientNet.from_pretrained('efficientnet-b2', num_classes=len(au_keys))
    }
    net = model_choices[args.model].to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    Contra_loss = ContrastiveLoss()
    O_loss = OrthogonalLoss()
    IDLoss = IDLoss()

    for epoch in range(args.num_epoch):
        train_loss = train(epoch, trainloader)

        writer = SummaryWriter(args.Descriptions)
        writer.add_scalars('losses', {'train_loss': train_loss}, global_step=epoch)

        if args.model == "Contra":

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'init_of_embed': net.init_of_embed
            }

        else:

            checkpoint = {'epoch': epoch, 'model_state_dict': net.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, f"{args.PATH_Checkpoint}{args.Descriptions}.pth")
