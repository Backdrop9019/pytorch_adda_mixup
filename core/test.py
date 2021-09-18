#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:03:50 2018

@author: gaoyi
"""

import torch
import torch.nn as nn

from utils import make_cuda


def eval_tgt(model, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    model.eval()
    # init loss and accuracy
    loss = 0
    acc = 0
    with torch.no_grad():
        # evaluate network
        for (images, labels) in data_loader:
            images = make_cuda(images)
            labels = make_cuda(labels).squeeze_()

            preds = model(images)
            _, preds = torch.max(preds.data, 1)
            acc += (preds == labels).float().sum()/images.shape[0]

        acc /= len(data_loader)

        print("Avg Accuracy = {:2%}".format(acc))