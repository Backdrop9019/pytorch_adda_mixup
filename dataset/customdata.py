from torchvision import transforms, datasets
import torch
import params

def get_custom(train,adp=False,size = 0):

    pre_process = transforms.Compose([transforms.Resize(params.image_size),
                                      transforms.ToTensor(),
  #  transforms.Normalize((0.5),(0.5)),
    ])
    custom_dataset = datasets.ImageFolder(
       root = params.custom_dataset_root ,
            transform = pre_process,
    )
    length = len(custom_dataset)
    train_set, val_set = torch.utils.data.random_split(custom_dataset, [int(length*0.9), length-int(length*0.9)])

    if train:
        train_set,_ = torch.utils.data.random_split(train_set, [size,len(train_set)-size])



    custom_data_loader = torch.utils.data.DataLoader(
        train_set if train else val_set,
        batch_size= params.adp_batch_size if adp else params.batch_size,
          shuffle=True,
        drop_last=True

    )

    return custom_data_loader