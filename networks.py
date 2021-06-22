import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from image_representation_learning.resnet import ResNetEncoder, ResNetBasicBlock

class ResNetEmbedding(nn.Module):
    def __init__(self):
        super(ResNetEmbedding, self).__init__()
        self.convnet = ResNetEncoder(3, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])

        self.fc = nn.Sequential(nn.Linear(512, 256),
                                nn.LeakyReLU(inplace=False),
                                nn.Linear(256, 128)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.LeakyReLU(inplace=False),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(inplace=False),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(inplace=False),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(128 * 4 * 4, 256),
                                nn.LeakyReLU(inplace=False),
                                nn.Linear(256, 128)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output = output / output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
