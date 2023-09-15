import torch
import torch.nn as nn


device = "cuda:0"


def generate_person_id_list():
    """Generates a list of person IDs. Female IDs are from F001 to F023, and Male IDs from M001 to M018."""

    # Format for the IDs - Female: F001 to F023, Male: M001 to M018
    female_ids = [f'F{str(i).zfill(3)}' for i in range(1, 24)]
    male_ids = [f'M{str(i).zfill(3)}' for i in range(1, 19)]

    # Return a combined list of female and male IDs
    return female_ids + male_ids


def get_clip(path_str):
    """Checks if any combination of person ID and clip name exists in the input string.
    If found, returns the combination; else returns None."""

    # Generate the list of people's IDs
    person_id_list = generate_person_id_list()

    # Generate a list of clip names (T1 to T8)
    clip_list = [f'T{i}' for i in range(1, 9)]

    # Generate all possible combinations of person ID and clip name
    all_paths = [f'{person_id}/{clip}' for person_id in person_id_list for clip in clip_list]

    # Check each path if it's a substring of the input string
    for path in all_paths:
        processed_path = f'{path[:5]}processed/{path[5:]}'
        if processed_path in path_str:
            return path  # Return the path if found

    return None  # Return None if no match is found


def get_frame_num(input_str):
    """Extracts the frame number from the end of a string, removing any leading zeros."""

    raw_frame_num = input_str[-10:-4]
    frame_num = raw_frame_num.lstrip('0')

    return frame_num

def SelectiveLearning_Uniform(labels):
    """Apply Selective Learning (SL) by assigning weights to the labels."""
    num_labels, num_classes = labels.shape
    SL_weights = torch.zeros_like(labels, dtype=torch.float32)

    for i in range(num_classes):
        class_labels = labels[:, i]
        ratio = class_labels.float().mean()

        if ratio > 0.5:
            # If the ratio is greater than 0.5
            SL_weights[class_labels == 1, i] = 1.0
            SL_weights[class_labels == 0, i] = 0.5 / (1 - ratio)

        elif ratio < 0.5 and ratio > 0:  # Ensure ratio is not zero to avoid division by zero
            # If the ratio is less than 0.5
            SL_weights[class_labels == 0, i] = 1.0
            SL_weights[class_labels == 1, i] = 0.5 / ratio

    return SL_weights.to(device)


def Get_ALL(outputs, targets):
    """Calculate True Positives, True Negatives, False Negatives, and False Positives."""
    # Apply sigmoid to output
    m = nn.Sigmoid()
    predicted = m(outputs)

    # Create a tensor where predictions >= 0.5 are 1, otherwise 0
    predicted_binary = (predicted >= 0.5).float()

    # Get True Positives: prediction and target are both 1
    AU_TP = (predicted_binary * targets).sum(dim=0).cpu().numpy()

    # Get True Negatives: prediction and target are both 0
    AU_TN = ((1 - predicted_binary) * (1 - targets)).sum(dim=0).cpu().numpy()

    # Get False Negatives: prediction is 0 but target is 1
    AU_FN = ((1 - predicted_binary) * targets).sum(dim=0).cpu().numpy()

    # Get False Positives: prediction is 1 but target is 0
    AU_FP = (predicted_binary * (1 - targets)).sum(dim=0).cpu().numpy()

    return AU_TP, AU_TN, AU_FN, AU_FP

def compute_metrics(TPs, TNs, FNs, FPs):
    """Compute precision, recall, F1-score, and accuracy."""
    epsilon = 1e-7
    precision = [TP / (TP + FP + epsilon) for TP, FP in zip(TPs, FPs)]
    recall = [TP / (TP + FN + epsilon) for TP, FN in zip(TPs, FNs)]
    F1 = [2 * r * p / (r + p + epsilon) for r, p in zip(recall, precision)]
    accuracy = [(TP + TN) / (TP + TN + FP + FN) for TP, TN, FP, FN in zip(TPs, TNs, FPs, FNs)]
    return precision, recall, F1, accuracy