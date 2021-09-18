

import os 
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import params
from dataset import get_mnist, get_mnist_m, get_usps,get_svhn,get_custom
import numpy as np
import itertools
import torch.nn.functional as F
import params

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
        #for REPRODUCIBILITY
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name,train=True,adp=False,size = 0):
    """Get data loader by name."""
    if name == "mnist":
        return get_mnist(train,adp,size)
    elif name == "mnist_m":
        return get_mnist_m(train,adp,size)
    elif name == "usps":
        return get_usps(train,adp,size)
    elif name == "svhn":
        return get_svhn(train,adp,size)
    elif name == "custom":
        return get_custom(train,adp,size)

def init_model(net, restore=None):
    """Init models with cuda and weights."""
    # init weights of model
    # net.apply(init_weights)

    print(f'restore file : {restore}')
    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        net.cuda()

    return net

def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self,smoothing):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    def forward(self, y, targets,smoothing=0.1):
        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(y, dim=-1) # 예측 확률 계산
        true_probs = torch.zeros_like(log_probs)
        true_probs.fill_(self.smoothing / (y.shape[1] - 1))
        true_probs.scatter_(1, targets.data.unsqueeze(1), confidence) # 정답 인덱스의 정답 확률을 confidence로 변경
        return torch.mean(torch.sum(true_probs * -log_probs, dim=-1)) # negative log likelihood

#mixup only data, not label
def mixup_data(source,target):
    max = params.lammax
    min = params.lammin
    lam = (max-min)*torch.rand((1))+min
    lam=lam.cuda()
    target = target.cuda()
    mixed_source = (1 - lam) * source +  lam* target
    

    return mixed_source,  lam




def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return 0.9* criterion(pred, y_a) + 0.1 * criterion(pred, y_b)