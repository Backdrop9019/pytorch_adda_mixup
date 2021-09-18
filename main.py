
import params
from utils import get_data_loader, init_model, init_random_seed,mixup_data
from core import pretrain , adapt , test,mixup
import torch
from models.models import *
import numpy as np
import sys



if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)
    print(f"Is cuda availabel? {torch.cuda.is_available()}")



    #set loader
    print("src data loader....")
    src_data_loader = get_data_loader(params.src_dataset,adp=False,size = 10000)
    src_data_loader_eval = get_data_loader(params.src_dataset,train=False)
    print("tgt data loader....")
    tgt_data_loader = get_data_loader(params.tgt_dataset,adp=False,size = 50000)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)
    print(f"scr data size : {len(src_data_loader.dataset)}")
    print(f"tgt data size : {len(tgt_data_loader.dataset)}")


    print("start training")
    source_cnn = CNN(in_channels=3).to("cuda")
    target_cnn = CNN(in_channels=3, target=True).to("cuda")
    discriminator = Discriminator().to("cuda")

    source_cnn = mixup.train_src(source_cnn, src_data_loader,tgt_data_loader,None)        
    # source_cnn.load_state_dict(torch.load("./generated/models/our-source_cnn-final.pt"))
    
    test.eval_tgt(source_cnn, tgt_data_loader_eval)

    target_cnn.load_state_dict(source_cnn.state_dict())
    
    tgt_encoder = mixup.train_tgt(source_cnn, target_cnn, discriminator,
                            src_data_loader,tgt_data_loader,None)



    print("=== Evaluating classifier for encoded target domain ===")
    print(f"mixup : {params.lammax} {params.src_dataset} -> {params.tgt_dataset} ")
    print("Eval | source_cnn | src_data_loader_eval")
    test.eval_tgt(source_cnn, src_data_loader_eval)
    print(">>> Eval | source_cnn | tgt_data_loader_eval <<<")
    test.eval_tgt(source_cnn, tgt_data_loader_eval)
    print(">>> Eval | target_cnn | tgt_data_loader_eval <<<")
    test.eval_tgt(target_cnn, tgt_data_loader_eval)




