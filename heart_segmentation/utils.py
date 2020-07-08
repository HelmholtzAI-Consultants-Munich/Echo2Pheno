'''
Code retrieved  from: https://github.com/ai-med/quickNAT_pytorch on 03/06/2020
'''

import torch
import numpy as np

def dice_confusion_matrix(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_cm = torch.zeros(num_classes, num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        for j in range(num_classes):
            Pred = (vol_output == j).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm