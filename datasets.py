import torch
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import os
import pandas as pd
from torchvision import transforms
import re
from utils import generate_person_id_list, get_clip, get_frame_num

# Define a common transformation structure to be used in different classes
TRANSFORM_TRAIN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(240),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5355, 0.4249, 0.3801), (0.2832, 0.2578, 0.2548)),
])

TRANSFORM_VAL = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.Normalize((0.5355, 0.4249, 0.3801), (0.2832, 0.2578, 0.2548)),
])


class BP4Ddataset(Dataset):
    # Initialization method for the dataset
    def __init__(self, homePath, datapath, labelpath, mode, fold, ID_required):
        self.ID_required = ID_required
        self.labels_path = labelpath
        self.mode = mode
        self.frameHome = homePath
        self.fold = fold
        self.transform_train = TRANSFORM_TRAIN
        self.transform_val = TRANSFORM_VAL

        if self.mode == "train":
            if self.fold == 1:
                self.frames_path = open(datapath + 'BP4D_combine_1_2_path.txt').readlines()
            elif self.fold == 2:
                self.frames_path = open(datapath + 'BP4D_combine_1_3_path.txt').readlines()
            elif self.fold == 3:
                self.frames_path = open(datapath + 'BP4D_combine_2_3_path.txt').readlines()

        elif self.mode == "val":
            if self.fold == 1:
                self.frames_path = open(datapath + 'BP4D_part3_path.txt').readlines()
            elif self.fold == 2:
                self.frames_path = open(datapath + 'BP4D_part2_path.txt').readlines()
            elif self.fold == 3:
                self.frames_path = open(datapath + 'BP4D_part1_path.txt').readlines()

        # Generate unique list of IDs from all frames
        if self.ID_required:
            self.IDs = [id for id in generate_person_id_list() if any(id in frame for frame in self.frames_path)]

    # Method to fetch an item from the dataset
    def __getitem__(self, index):
        framePath = self.frameHome + self.frames_path[index].strip()
        frame = Image.open(framePath).convert('RGB') if Image.open(framePath).mode != 'RGB' else Image.open(framePath)
        frame = self.transform_train(frame) if self.mode == "train" else self.transform_val(frame)

        # Find the labels
        clip = get_clip(framePath).replace('/', '_')
        frame_num = get_frame_num(framePath)
        path = glob(os.path.join(self.labels_path + '*.csv'))

        for i in path:
            if clip in i:
                df = pd.read_csv(i, header=None)
                label = df.values[int(frame_num), :]

        tensor_label = torch.Tensor(label).to(torch.float32)
        if self.ID_required:
            img_ID = clip.split('_')[0]
            return frame, img_ID, tensor_label
        else:
            return frame, tensor_label

    # Method to get the length of the dataset
    def __len__(self):
        return len(self.frames_path)


class DISFAdataset(Dataset):
    def __init__(self, homePath, datapath, labelpath, mode, fold, ID_required):
        self.ID_required = ID_required
        self.labels_path = labelpath
        self.mode = mode
        self.frameHome = homePath
        self.fold = fold
        self.transform_train = TRANSFORM_TRAIN
        self.transform_val = TRANSFORM_VAL

        # Get the correct data file based on mode and fold
        if self.mode == "val":
            self.frames_path = open(datapath + f'DISFA_test_img_path_fold{fold}.txt').readlines()
        else:
            self.frames_path = open(datapath + f'DISFA_train_img_path_fold{fold}.txt').readlines()

        # Generate unique list of IDs from all frames
        if self.ID_required:
            pattern_3 = r"\bSN\d{3}\b"
            self.IDs = list({id for frame in self.frames_path for id in re.findall(pattern_3, frame)})

    def take_FrameNum(self, path):
        pattern = r'\d{6}'
        match = re.search(pattern, path)
        return int(match.group()) if match else None

    def __getitem__(self, index):
        framePath = self.frameHome + self.frames_path[index].strip()
        frame = Image.open(framePath).convert('RGB') if Image.open(framePath).mode != 'RGB' else Image.open(framePath)
        frame = self.transform_train(frame) if self.mode == "train" else self.transform_val(frame)

        frame_num = self.take_FrameNum(framePath)
        # take the frame number and find the label
        pattern_3 = r"\bSN\d{3}\b"

        ID = re.findall(pattern_3, framePath)

        path = glob(os.path.join(self.labels_path + ID[0] + '/' + '*.txt'))

        labels = torch.zeros(8)
        au_keys = ['au1.txt', 'au2.txt', 'au4.txt', 'au6.txt', 'au9.txt', 'au12.txt', 'au25.txt', 'au26.txt']

        for k in range(len(au_keys)):
            for i in range(len(path)):
                if au_keys[k] in path[i]:
                    with open(path[i]) as f:
                        lines = f.readlines()
                        if int(lines[frame_num - 1].split(',')[1]) >= 2:
                            labels[k] = 1
        if self.ID_required:
            return frame, ID[0], labels
        else:
            return frame, labels

    def __len__(self):
        return len(self.frames_path)
