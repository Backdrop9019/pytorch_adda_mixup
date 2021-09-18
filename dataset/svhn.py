import torch
from torchvision import datasets, transforms

import params

import torch.utils.data as data_utils

def get_svhn(train,adp=False,size=0):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([
    transforms.Resize(params.image_size),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   
   ])


    # dataset and data loader
    svhn_dataset = datasets.SVHN(root=params.svhn_dataset_root,
                                   split='train' if train else 'test',
                                   transform=pre_process,
                                   download=True)
    if train:
        # perm = torch.randperm(len(svhn_dataset))
        # indices = perm[:10000]
        svhn_dataset,_ = data_utils.random_split(svhn_dataset, [size,len(svhn_dataset)-size])
        # size = len(svhn_dataset)
        # train, valid = data_utils.random_split(svhn_dataset,[size-int(size*params.train_val_ratio),int(size*params.train_val_ratio)])
    
        # train_loader = torch.utils.data.DataLoader(
        #     dataset=train,
        #     batch_size= params.adp_batch_size if adp else params.batch_size,

        #     shuffle=True,
        #     drop_last=True)

        # valid_loader = torch.utils.data.DataLoader(
        #     dataset=valid,
        #     batch_size= params.adp_batch_size if adp else params.batch_size,

        #     shuffle=True,
        #     drop_last=True)
        # return train_loader,valid_loader

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size= params.adp_batch_size if adp else params.batch_size,

        shuffle=True,
        drop_last=True)

    return svhn_data_loader