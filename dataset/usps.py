import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

import params
import torch.utils.data as data_utils


def get_usps(train,adp=False,size=0):
    """Get usps dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(params.image_size),
                                      transforms.ToTensor(),
        # transforms.Normalize((0.5),(0.5)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                    #   transforms.Grayscale(1),


        
        ])


    # dataset and data loader
    usps_dataset = datasets.USPS(root=params.usps_dataset_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)


    if train:
        usps_dataset,  _   = data_utils.random_split(usps_dataset, [size,len(usps_dataset)-size])
        # size = len(usps_dataset)
        # train, valid = data_utils.random_split(usps_dataset,[size-int(size*params.train_val_ratio),int(size*params.train_val_ratio)])
        # train_loader = torch.utils.data.DataLoader(
        # dataset=train,
        # batch_size= params.adp_batch_size if adp else params.batch_size,
        # shuffle=True,
        # drop_last=True)
        # valid_loader = torch.utils.data.DataLoader(
        # dataset=valid,
        # batch_size= params.adp_batch_size if adp else params.batch_size,
        # shuffle=True,
        # drop_last=True)
        # return train_loader,valid_loader

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size= params.adp_batch_size if adp else params.batch_size,

        shuffle=True,
        drop_last=True)
    return usps_data_loader