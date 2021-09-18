from torch import nn
import torch.nn.functional as F
import params

class Encoder(nn.Module):
    def __init__(self, in_channels=1, h=256, dropout=0.5):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout =nn.Dropout2d(p= dropout)
        # self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(800, 500)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        bs = x.size(0)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.dropout(self.conv2(x)))))
        x = x.view(bs, -1)
        # x = self.dropout(x)W
        x = self.fc1(x)
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(500, n_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.l1(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10, target=False):
        super(CNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.classifier = Classifier(n_classes)
        if target:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, h=500):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(500, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 2)
        # self.slope =params.slope
        
        self.relu = nn.ReLU()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x
