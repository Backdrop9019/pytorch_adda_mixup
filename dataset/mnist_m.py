import torch.utils.data as data
from PIL import Image
import os
import params
from torchvision import transforms
import torch 

import torch.utils.data as data_utils


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data_ in data_list:
            self.img_paths.append(data_[:-3])
            self.img_labels.append(data_[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


def get_mnist_m(train,adp=False,size= 0 ):

    if train == True:
        mode = 'train'
    else:
        mode = 'test'

    train_list = os.path.join(params.mnist_m_dataset_root, 'mnist_m_{}_labels.txt'.format(mode))
        # image pre-processing
    pre_process = transforms.Compose([
        transforms.Resize(params.image_size),
    # transforms.Grayscale(3),

                                      transforms.ToTensor(),                                     

#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#    transforms.Grayscale(1),
   ]
   )

    dataset_target = GetLoader(
        data_root=os.path.join(params.mnist_m_dataset_root, 'mnist_m_{}'.format(mode)),
        data_list=train_list,
        transform=pre_process)
        
    if train:
        # perm = torch.randperm(len(dataset_target))
        # indices = perm[:10000]
        dataset_target,_ = data_utils.random_split(dataset_target, [size,len(dataset_target)-size])
        # size = len(dataset_target)
        # train, valid = data_utils.random_split(dataset_target,[size-int(size*params.train_val_ratio),int(size*params.train_val_ratio)])        
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


    dataloader  = torch.utils.data.DataLoader(
        dataset=dataset_target,
                batch_size= params.adp_batch_size if adp else params.batch_size,

        shuffle=True,
        drop_last=True)
    
    return dataloader