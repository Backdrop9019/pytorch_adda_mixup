import torch.nn as nn
import torch
import torch.optim as optim

import params
from utils import make_cuda, save_model, LabelSmoothingCrossEntropy,mixup_data,mixup_criterion
from random import *
import sys

from torch.utils.data import Dataset,DataLoader
import os
from core import test
from utils import make_cuda, mixup_data



class CustomDataset(Dataset): 
  def __init__(self,img,label):
    self.x_data = img
    self.y_data = label
  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx]
    y = self.y_data[idx]
    return x, y


def train_src(model, source_data_loader,target_data_loader,valid_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    model.train()


    target_data_loader = list(target_data_loader)

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
    len_data_loader = min(len(source_data_loader), len(target_data_loader))

    for epoch in range(params.num_epochs_pre+1):
        data_zip = enumerate(zip(source_data_loader, target_data_loader))
        for step, ((images, labels), (images_tgt, _)) in data_zip:
            # make images and labels variable
            images = make_cuda(images)
            labels = make_cuda(labels.squeeze_())
            # zero gradients for optimizer
            optimizer.zero_grad()
            target=target_data_loader[randint(0, len(target_data_loader)-1)]
            images, lam = mixup_data(images,target[0])

            # compute loss for critic
            preds = model(images)
            # loss = mixup_criterion(criterion, preds, labels, labels_tgt, lam)
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()
            


      # # eval model on test set
        if ((epoch) % params.eval_step_pre == 0):
            print(f"Epoch [{epoch}/{params.num_epochs_pre}]",end='')
            if valid_loader is not None:
                test.eval_tgt(model, valid_loader)
            else:
                test.eval_tgt(model, source_data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(model, "our-source_cnn-{}.pt".format(epoch + 1))

    # # save final model
    save_model(model, "our-source_cnn-final.pt")

    return model




def train_tgt(source_cnn, target_cnn, critic,
              src_data_loader, tgt_data_loader,valid_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    source_cnn.eval()
    target_cnn.encoder.train()
    critic.train()
    isbest = 0
    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    #target encoder
    optimizer_tgt = optim.Adam(target_cnn.parameters(),
                               lr=params.adp_c_learning_rate,
                               betas=(params.beta1, params.beta2),
                               weight_decay=params.weight_decay
                               )
    #Discriminator
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2),
                               weight_decay=params.weight_decay

                                  
                                  )

    ####################
    # 2. train network #
    ####################
    data_len = min(len(src_data_loader), len(tgt_data_loader))

    for epoch in range(params.num_epochs+1):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:

            # make images variable
            images_src = make_cuda(images_src)
            images_tgt = make_cuda(images_tgt)

            ###########################
            # 2.1 train discriminator #
            ###########################

            # mixup  data
            images_src, _ = mixup_data(images_src,images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = source_cnn.encoder(images_src)
            feat_tgt = target_cnn.encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = make_cuda(torch.zeros(feat_src.size(0)).long())
            label_tgt = make_cuda(torch.ones(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()


            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = target_cnn.encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()
            #######################
            # 2.3 print step info #
            #######################
            if ((epoch % 10 ==0 )& ((step + 1) %  data_len == 0)):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch ,
                              params.num_epochs,
                              step + 1,
                              data_len,
                              loss_critic.item(),
                              loss_tgt.item(),
                              acc.item()))
                if valid_loader is not None:
                    test.eval_tgt(target_cnn,valid_loader)

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "our-critic-final.pt"))
    torch.save(target_cnn.state_dict(), os.path.join(
        params.model_root,
        "our-target_cnn-final.pt"))
    return target_cnn