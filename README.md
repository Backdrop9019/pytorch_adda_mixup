# PyTorch-ADDA-mixup
A PyTorch implementation added MIXUO for Adversarial Discriminative Domain Adaptation.

Confirmed improved performance by mixing up target domian and source domain

# Usage
It works on MNIST -> USPS , SVHN -> MNIST , USPS -> MNIST, MNIST -> MNIST-M
Only 10,000 of the total data were used.(usps excluded)

<pre>
<code>
python main.py
</code>
</pre>

## adda
This repo is  based on https://github.com/corenel/pytorch-adda  , https://github.com/Fujiki-Nakamura/ADDA.PyTorch



Reference  
https://arxiv.org/abs/1702.05464  


# 
