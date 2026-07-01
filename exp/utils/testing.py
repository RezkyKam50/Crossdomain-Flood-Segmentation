import matplotlib.pyplot as plt
import torchmetrics as TM
import torch
from torch import nn
import torchvision
import torchvision.transforms as T
import os


t2img = T.ToPILImage()
img2t = T.ToTensor()

def computeIOU(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  intersection = torch.sum(output * target)
  union = torch.sum(target) + torch.sum(output) - intersection
  iou = (intersection + .0000001) / (union + .0000001)
  
  if iou != iou:
    print("failed, replacing with 0")
    iou = torch.tensor(0).float()
  
  return iou

def computeMetrics(output, target, device, criterion):
  loss = criterion(output, target.squeeze(1).long())
  TP = truePositives(output, target, device).float()
  FN = falseNegatives(output, target, device).float()
  FP = falsePositives(output, target, device).float()
  TN = trueNegatives(output, target, device).float()
  
  return {
      "TP": TP,
      "FN": FN,
      "FP": FP,
      "TN": TN,
      "loss": loss.item()
  }

def computeAccuracy(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  correct = torch.sum(output.eq(target))
  
  return correct.float() / len(target)

def truePositives(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  correct = torch.sum(output * target)
  
  return correct

def trueNegatives(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  output = (output == 0)
  target = (target == 0)
  correct = torch.sum(output * target)
  
  return correct

def falsePositives(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  output = (output == 1)
  target = (target == 0)
  correct = torch.sum(output * target)
  
  return correct

def falseNegatives(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  output = (output == 0)
  target = (target == 1)
  correct = torch.sum(output * target)
  
  return correct

from enum import IntEnum
class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    
