import torch.nn as nn
import torch.optim as optim
import torch
import params
from utils import make_cuda, save_model, LabelSmoothingCrossEntropy,mixup_data
from random import *
import sys

def train_src(model, source_data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    model.train()



    # setup criterion and optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=params.pre_c_learning_rate,
        betas=(params.beta1, params.beta2),
        weight_decay=params.weight_decay
        )


    if params.labelsmoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing= params.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()


    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(source_data_loader):
            # make images and labels variable
            images = make_cuda(images)
            labels = make_cuda(labels.squeeze_())
            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = model(images)
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()
            


        # # eval model on test set
        if ((epoch ) % params.eval_step_pre == 0):
            print(f"Epoch [{epoch}/{params.num_epochs_pre}]",end='')
            eval_src(model, source_data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(model, "ADDA-source_cnn-{}.pt".format(epoch + 1))

    # # save final model
    save_model(model, "ADDA-source_cnn-final.pt")

    return model

def eval_src(model, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    model.eval()
    with torch.no_grad():
        # init loss and accuracy
        loss = 0
        acc = 0

        # evaluate network
        for (images, labels) in data_loader:
            
            images = make_cuda(images)
            labels = make_cuda(labels).squeeze_()

            preds = model(images)

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).cpu().sum().item()

        acc /= len(data_loader.dataset)

        print("Avg Accuracy = {:2%}".format( acc))


