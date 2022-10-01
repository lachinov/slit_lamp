import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Mix(nn.Module):
    def __init__(self, losses, coefficients=None):
        super(Mix, self).__init__()
        self.losses = losses
        self.coefficients = coefficients

        if self.coefficients is None:
            self.coefficients = { k:1 for k in self.losses}

    def forward(self, target, predict):

        losses_results = {k:self.losses[k](target, predict) for k in self.losses }

        loss = sum([losses_results[k]*self.coefficients[k] for k in losses_results if losses_results[k] is not None]) / (len(losses_results))

        return loss, losses_results

class LossWrapper(nn.Module):
    def __init__(self,loss, output_key=0, target_key=0):
        super(LossWrapper, self).__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.counter = 0
        self.loss = loss

    def forward(self, target, predict):

        pred = predict[self.output_key]
        gt = target[self.target_key]

        loss = self.loss(pred,gt)

        self.counter += 1

        return loss

class PrecisionLoss(nn.Module):
    def __init__(self,output_key=0, target_key=0):
        super(PrecisionLoss, self).__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.counter = 0

    def forward(self, target, predict):

        pred = predict[self.output_key]
        gt = target[self.target_key]

        tp = (gt*pred).sum()

        loss = 1 - (tp+1e-2)/(pred.sum()+1e-2)#self.loss(pred,gt)

        self.counter += 1

        return loss


class RecallLoss(nn.Module):
    def __init__(self,output_key=0, target_key=0):
        super(RecallLoss, self).__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.counter = 0

    def forward(self, target, predict):

        pred = predict[self.output_key]
        gt = target[self.target_key]

        tp = (gt*pred).sum()
        fn = (gt*(1-pred)).sum()

        loss = 1 - (tp+1e-2)/(tp+fn+1e-2)#self.loss(pred,gt)

        self.counter += 1

        return loss