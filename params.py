import torch

# params for dataset and data loader
data_root = "data"
image_size = 28

#restore
model_root = 'generated\\models'


# params for target dataset
# 'mnist_m', 'usps', 'svhn' "custom"

#dataset root
mnist_dataset_root = data_root
mnist_m_dataset_root = data_root+'\\mnist_m'
usps_dataset_root = data_root+'\\usps'
svhn_dataset_root = data_root+'\\svhn'
custom_dataset_root = data_root+'\\custom\\'

# params for training network
num_gpu = 1

log_step_pre = 10
log_step = 10
eval_step_pre = 10

##epoch
save_step_pre = 100
manual_seed = 1234

d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = 'generated\\models\\ADDA-critic-final.pt'

## sorce target
src_dataset = 'custom'
tgt_dataset = 'custom'


# params for optimizing models
# # lam 0.3
#mnist -> custom
num_epochs_pre = 20
num_epochs = 50
batch_size = 128
adp_batch_size =  128
pre_c_learning_rate = 2e-4
adp_c_learning_rate = 1e-4
d_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.999
weight_decay = 0


# #usps -> custom
# #lam 0.1
# num_epochs_pre = 5
# num_epochs = 20
# batch_size = 256
# pre_c_learning_rate = 1e-4
# adp_c_learning_rate = 2e-5
# d_learning_rate = 1e-5
# beta1 = 0.5
# beta2 = 0.999
# weight_decay = 2e-4

# #mnist_m -> custom
# #lam 0.1
# num_epochs_pre = 30
# num_epochs = 50
# batch_size = 256
# adp_batch_size =  256
# pre_c_learning_rate = 1e-3
# adp_c_learning_rate = 1e-4
# d_learning_rate = 1e-4
# beta1 = 0.5
# beta2 = 0.999
# weight_decay = 2e-4

# # params for optimizing models
#lam 0.3
# #mnist -> mnist_m
# num_epochs_pre = 50
# num_epochs = 100
# batch_size = 256
# adp_batch_size =  256
# pre_c_learning_rate = 2e-4
# adp_c_learning_rate = 2e-4
# d_learning_rate = 2e-4
# beta1 = 0.5
# beta2 = 0.999
# weight_decay = 0

# # source 10000 target 50000
# # params for optimizing models
# #svhn -> mnist
# num_epochs_pre = 20
# num_epochs = 30
# batch_size = 128
# adp_batch_size =  128
# pre_c_learning_rate = 2e-4
# adp_c_learning_rate = 1e-4
# d_learning_rate = 1e-4
# beta1 = 0.5
# beta2 = 0.999
# weight_decay = 2.5e-4

# # mnist->usps
# num_epochs_pre = 50
# num_epochs = 100
# batch_size = 256
# adp_batch_size = 256
# pre_c_learning_rate = 2e-4
# adp_c_learning_rate = 2e-4
# d_learning_rate = 2e-4
# beta1 = 0.5
# beta2 = 0.999
# weight_decay =0


# # usps->mnist
# num_epochs_pre = 50
# num_epochs = 100
# batch_size = 256
# pre_c_learning_rate = 2e-4
# adp_c_learning_rate = 2e-4
# d_learning_rate =2e-4
# beta1 = 0.5
# beta2 = 0.999
# weight_decay =0



#
use_load = False
train =False

#ratio mix target
lammax = 0.0
lammin = 0.0


labelsmoothing = False
smoothing = 0.3