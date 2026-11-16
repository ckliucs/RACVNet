"""Complementary spatial and channel fusion with a bounded residual."""
import math
import torch
from torch import nn
from torch.cuda.amp import autocast


class ContextDrivenRefinementFusion(nn.Module):
    def __init__(self,channels=192,hidden=32,maximum_relative=.1,initial_strength=.02):
        super().__init__()
        if not 0<initial_strength<maximum_relative:raise ValueError('Invalid residual strength')
        self.channels,self.maximum_relative=channels,float(maximum_relative)
        self.shared_projection=nn.Sequential(nn.Conv2d(channels,hidden,1,bias=False),
            nn.GroupNorm(8,hidden),nn.GELU())
        self.detail_saliency=nn.Conv2d(hidden,1,1)
        self.semantic_saliency=nn.Conv2d(hidden,1,1)
        self.semantic_channels=nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden,hidden//4,1),nn.GELU(),nn.Conv2d(hidden//4,hidden,1),nn.Sigmoid())
        self.mix=nn.Sequential(nn.Conv2d(hidden*2,hidden,1,bias=False),
            nn.GroupNorm(8,hidden),nn.GELU(),
            nn.Conv2d(hidden,hidden,3,padding=1,groups=hidden,bias=False),nn.GELU())
        # No channel-constant bias; directional task signal exists from step1.
        self.output_projection=nn.Conv2d(hidden,channels,1,bias=False)
        self.raw_gain=nn.Parameter(torch.tensor(math.log(initial_strength/(maximum_relative-initial_strength))))
        self.last_diagnostics={}

    def forward(self,semantic,detail,anchor):
        if semantic.shape!=detail.shape or anchor.shape!=detail.shape or detail.ndim!=4 or detail.shape[1]!=self.channels:
            raise ValueError('CDRF requires matching same-view BCHW feature maps')
        with autocast(enabled=False):
            # Identical learned channel basis and normalization for both inputs.
            low=self.shared_projection(detail.float())
            high=self.shared_projection(semantic.float())
            low_attention=self.detail_saliency(low).sigmoid()
            high_attention=self.semantic_saliency(high).sigmoid()
            low_cross,high_cross=low*high_attention,high*low_attention
            common=(low_cross+high_cross)*.5
            channel_weight=2.*self.semantic_channels(high)
            complementary=(low_cross-high_cross)*channel_weight
            raw=self.output_projection(self.mix(torch.cat((common,complementary),dim=1)))
            centered=raw-raw.mean((2,3),keepdim=True)
            raw_rms=centered.detach().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
            direction=torch.tanh(centered/raw_rms)
            # Tanh may reintroduce a nonzero spatial mean. Recenter and enforce
            # |direction|<=1 without depending on the arbitrary raw amplitude.
            direction=direction-direction.mean((2,3),keepdim=True)
            peak=direction.detach().abs().amax((1,2,3),keepdim=True).clamp_min(1.)
            direction=direction/peak
            reference=anchor.detach().float().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
            strength=self.maximum_relative*self.raw_gain.sigmoid()
            correction=strength*reference*direction
            result=anchor+correction.to(anchor.dtype)
            with torch.no_grad():
                self.last_diagnostics={
                    'strength':float(strength),'raw_centered_rms':float(raw_rms.mean()),
                    'correction_rms':float(correction.square().mean().sqrt()),
                    'correction_relative_rms':float((correction.square().mean((1,2,3),keepdim=True).sqrt()/reference).max()),
                    'correction_spatial_mean_abs':float(correction.mean((2,3)).abs().max()),
                    'detail_saliency_mean':float(low_attention.mean()),
                    'semantic_saliency_mean':float(high_attention.mean()),
                    'semantic_channel_mean':float(channel_weight.mean()),
                    'common_rms':float(common.square().mean().sqrt()),
                    'complementary_rms':float(complementary.square().mean().sqrt())}
            return result
