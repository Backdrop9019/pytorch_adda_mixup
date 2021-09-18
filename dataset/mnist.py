

import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils

import params


def get_mnist(train,adp = False,size = 0):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(params.image_size),
                                      transforms.ToTensor(),
#    transforms.Normalize((0.5),(0.5)),
           transforms.Lambda(lambda x: x.repeat(3, 1, 1)),


   ])
                                  



    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.mnist_dataset_root,
                                   train=train,
                                   transform=pre_process,
                                   
                                   download=True)
    if train:
        # perm = torch.randperm(len(mnist_dataset))
        # indices = perm[:10000]
        mnist_dataset,_ = data_utils.random_split(mnist_dataset, [size,len(mnist_dataset)-size])
        # size = len(mnist_dataset)
        # train, valid = data_utils.random_split(mnist_dataset,[size-int(size*params.train_val_ratio),int(size*params.train_val_ratio)])
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

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size= params.adp_batch_size if adp else params.batch_size,
        shuffle=True,
        drop_last=True)
    return mnist_data_loader