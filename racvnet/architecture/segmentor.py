"""Stage-I segmentation network combining RAM and CDRF on CASCADE."""
import torch
from .cascade_dependency import translate_decoder_state
from .ram_segmentor import RegionAwareSegmentor
from racvnet.cdrf import ContextDrivenRefinementFusion


class RACVNetSegmentor(RegionAwareSegmentor):
    def __init__(self,weights,pretrained=False):
        super().__init__(weights,pretrained=pretrained)
        with torch.random.fork_rng(devices=[]):
            # Keep "sdcf" as the registered module name for checkpoint compatibility.
            self.sdcf=ContextDrivenRefinementFusion(channels=192,hidden=32,maximum_relative=.1,initial_strength=.02)
        self._cdrf_records=[]
        gate = self.decoder.AG2
        forward = gate.forward

        def refine_gate(g, x):
            # CASCADE invokes AG2 with g= and x=, including on PyTorch 1.9.
            return self._refine_skip(g, x, forward(g, x))

        gate.forward = refine_gate

    def load_state_dict(self,state_dict,strict=True):
        # Torch1.9 silently accepts a vector for a scalar parameter by taking
        # element0. Reject incomplete/incompatible tensors before native loading.
        if not strict:raise RuntimeError('RACVNet requires strict checkpoint loading')
        state_dict = translate_decoder_state(state_dict)
        expected=self.state_dict()
        if set(expected)!=set(state_dict):
            raise RuntimeError('RACVNet checkpoint key set differs: missing=%s unexpected=%s'%
                (sorted(set(expected)-set(state_dict)),sorted(set(state_dict)-set(expected))))
        for name,value in state_dict.items():
            if not isinstance(value,torch.Tensor) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype:
                raise RuntimeError('RACVNet checkpoint shape/dtype mismatch: '+name)
        return super().load_state_dict(state_dict,strict=True)

    def _refine_skip(self,semantic,detail,output):
        value=self.sdcf(semantic,detail,output)
        self._cdrf_records.append(dict(self.sdcf.last_diagnostics))
        return value

    def forward(self,images,return_aux=False):
        self._cdrf_records=[]
        output=super().forward(images,return_aux=True)
        if len(self._cdrf_records)!=2:
            raise RuntimeError('Exactly one CDRF call is required for each RAM view')
        if not return_aux:return output['seg']
        output['cdrf_diagnostics']={view+'_'+key:value
            for view,record in zip(('global','local'),self._cdrf_records)
            for key,value in record.items()}
        return output


def build(weights,pretrained=False):
    return RACVNetSegmentor(weights,pretrained=pretrained)
