"""Small residual 3D refinement. Zero output projection preserves input logits."""
import torch
from torch import nn
from torch.nn import functional as F


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv3d(13,16,3,padding=1),nn.GroupNorm(4,16),nn.ReLU())
        self.down=nn.Sequential(nn.Conv3d(16,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.ReLU(),
            nn.Conv3d(32,32,3,padding=1,groups=32),nn.Conv3d(32,32,1),nn.GroupNorm(8,32),nn.ReLU())
        self.fuse=nn.Sequential(nn.Conv3d(48,16,1),nn.GroupNorm(4,16),nn.ReLU(),
            nn.Conv3d(16,16,3,padding=1,groups=16),nn.Conv3d(16,16,1),nn.GroupNorm(4,16),nn.ReLU())
        self.head=nn.Conv3d(16,9,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,probability,image,coordinates):
        assert probability.ndim==5 and probability.shape[1]==9
        assert image.shape==(probability.shape[0],1,*probability.shape[2:])
        assert coordinates.shape==(probability.shape[0],3,*probability.shape[2:])
        local=self.stem(torch.cat((probability,image,coordinates),1))
        global_context=F.interpolate(self.down(local),size=local.shape[2:],mode='trilinear',align_corners=False)
        delta=self.head(self.fuse(torch.cat((local,global_context),1)))
        return probability.clamp_min(1e-6).log()+delta


def objective(logits,target):
    ce=F.cross_entropy(logits,target)
    p=logits.softmax(1)
    g=F.one_hot(target,9).permute(0,4,1,2,3).float()
    dims=(0,2,3,4)
    dice=1-((2*(p*g).sum(dims)+1e-5)/(p.square().sum(dims)+g.square().sum(dims)+1e-5)).mean()
    return .5*ce+.5*dice,ce,dice
