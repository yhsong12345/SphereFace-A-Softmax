import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *



class SphereFace(nn.Module):
    def __init__(self, num_layers=4, num_classes=10):
        super(SphereFace, self).__init__()

        if num_layers == 4:
            n = [0, 0, 0, 0]
        elif num_layers == 10:
            n = [0, 1, 2, 1]
        elif num_layers == 20:
            n = [1, 2, 4, 1]
        elif num_layers == 36:
            n = [2, 4, 8, 2]
        elif num_layers == 64:
            n = [3, 8, 16, 3]


        filters = [1, 64, 128, 256, 512]

        self.layer = self._make_layers(ResBlock, filters, n, stride=2)
        self.FC1 = nn.Linear(512*2*2, 512)
        self.FC2 = Anglelayer(512, num_classes)

        #weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)


    
    def _make_layers(self, block, filters, layer, stride):
        layers = []
        for i in range(len(layer)):
            layers.append(conv(filters[i], filters[i+1], stride=stride))
            ns = [1]*(layer[i])
            if ns:
                for s in ns:
                    layers.append(block(filters[i+1], s))
            
        return nn.Sequential(*layers)
        

    def forward(self, x):
        o = self.layer(x)
        o = o.view(o.size(0), -1)
        o = self.FC1(o)
        r = self.FC2(o)
        return o, r
    


class A_Softamx(nn.Module):
    def __init__(self):
        super(A_Softamx, self).__init__()
        self.lamb = 1500.0
        self.lmbMax = 1500.0
        self.lambMin = 5
        self.iter = 0


    def forward(self, feature, target):
        self.iter += 1
        cos_theta, phi_theta = feature
        target = target.view(-1,1)

        index = cos_theta.data * 0.0
        index.scatter_(1, target.data.view(-1,1),1)
        index = index.byte()
        index = torch.autograd.Variable(index)

        output = cos_theta * 1.0

        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -logpt

        return loss.mean()