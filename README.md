# PyTorch-ADDA-mixup
A PyTorch implementation added MIXUP for Adversarial Discriminative Domain Adaptation.

Confirmed improved performance by mixing up target domian and source domain

# Usage
It works on MNIST -> USPS , SVHN -> MNIST , USPS -> MNIST, MNIST -> MNIST-M
Only 10,000 of the total data were used.(usps excluded)

<pre>
<code>
python main.py
</code>
</pre>

# Result

![image](https://user-images.githubusercontent.com/52914552/133892466-99846090-90be-45da-8bbb-f4de89372c50.png)


## Reference
This repo is  based on https://github.com/corenel/pytorch-adda  , https://github.com/Fujiki-Nakamura/ADDA.PyTorch
https://arxiv.org/abs/1702.05464  


