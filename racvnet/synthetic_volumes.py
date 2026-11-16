"""GT-based corruption from all18 training patients. No test path is accepted."""
from collections import OrderedDict
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset
from .protocol import TRAIN_CASES as TRAIN, LVRM_SETTINGS as SETTINGS


def coordinates(shape):
    values=[np.linspace(-1,1,n,dtype=np.float32) if n>1 else np.zeros(1,np.float32) for n in shape]
    return np.stack(np.broadcast_arrays(values[0][:,None,None],values[1][None,:,None],values[2][None,None,:])).copy()


def probability_from_mask(mask,rng):
    channels=[]
    for k in range(9):
        channel=ndi.gaussian_filter((mask==k).astype(np.float32),sigma=SETTINGS['probability_smoothing_sigma'])
        channels.append(channel)
    p=np.stack(channels)
    confidence=rng.uniform(*SETTINGS['probability_confidence_range'])
    p=confidence*p+(1-confidence)/9
    p/=p.sum(0,keepdims=True)
    return p.astype(np.float32)


def ellipsoid(shape,center,radius):
    spans=tuple(slice(max(0,int(c-r)),min(n,int(c+r+1))) for n,c,r in zip(shape,center,radius))
    grid=np.ogrid[tuple(slice(s.start,s.stop) for s in spans)]
    inside=sum(((axis-c)/r)**2 for axis,c,r in zip(grid,center,radius))<=1
    return spans,inside


def corrupt(target,full_gt,origin,rng,force_modes=None):
    noisy=target.copy();events=[]
    present=np.unique(target);present=present[present>0]
    if force_modes is None and rng.rand()<SETTINGS['synthetic_clean_probability']:
        return noisy,['clean']
    modes=force_modes if force_modes is not None else rng.choice(SETTINGS['synthetic_modes'],size=int(rng.randint(1,3)),replace=False)
    for mode in modes:
        if mode=='remote_fp':
            available=np.unique(full_gt);available=available[available>0]
            if not len(available):continue
            k=int(rng.choice(available))
            whole_points=np.argwhere(full_gt==k)
            if not len(whole_points):continue
            tree=cKDTree(whole_points)
            chosen=None
            radius=np.array([rng.randint(1,4),rng.randint(4,11),rng.randint(4,11)])
            for _ in range(30):
                center=np.array([rng.randint(r,n-r) for n,r in zip(target.shape,radius)])
                spans,region=ellipsoid(target.shape,center,radius)
                local_points=np.argwhere(region)+np.array([s.start for s in spans])
                # Verify every synthetic FP voxel against full-patient target GT.
                distances=tree.query(local_points+np.array(origin),eps=0,workers=1)[0]
                if distances.min()>=SETTINGS['synthetic_remote_min_distance'] and np.all(target[spans][region]==0):
                    chosen=(spans,region);break
            if chosen:
                spans,region=chosen;noisy[spans][region]=k
                events.append('remote_fp')
        elif mode=='slice_gap' and len(present):
            k=int(rng.choice(present));zs=np.flatnonzero((target==k).any((1,2)))
            z=int(rng.choice(zs));width=int(rng.randint(1,5))
            slab=noisy[z:min(len(noisy),z+width)]
            slab[slab==k]=0;events.append('slice_gap')
        elif mode=='confusion' and len(present):
            k=int(rng.choice(present));other=(7-k) if k in (3,4) else int(rng.choice([v for v in range(1,9) if v!=k]))
            points=np.argwhere(target==k);center=points[rng.randint(len(points))]
            spans,region=ellipsoid(target.shape,center,(3,12,12))
            selected=region&(noisy[spans]==k);noisy[spans][selected]=other
            events.append('confusion')
        elif mode=='boundary' and len(present):
            k=int(rng.choice(present));mask=noisy==k
            if rng.rand()<.5:
                enlarged=ndi.binary_dilation(mask,iterations=int(rng.randint(1,4)))
                noisy[enlarged&(noisy==0)]=k
            else:
                eroded=ndi.binary_erosion(mask,iterations=int(rng.randint(1,3)))
                noisy[mask&~eroded]=0
            events.append('boundary')
    return noisy,events or ['unchanged_after_rejected_corruption']


class SyntheticVolumes(Dataset):
    def __init__(self, epoch, cache_dir):
        self.epoch = epoch
        self.cache_dir = cache_dir
        self.cache = OrderedDict()
    def __len__(self):return len(TRAIN)*SETTINGS['patches_per_patient']
    def __getitem__(self,index):
        patient=TRAIN[index//SETTINGS['patches_per_patient']]
        if patient not in self.cache:
            with np.load(self.cache_dir / (patient + '.npz')) as f:self.cache[patient]=(f['image'],f['label'])
            if len(self.cache)>2:self.cache.popitem(last=False)
        image,gt=self.cache[patient]
        rng=np.random.RandomState(SETTINGS['seed']+self.epoch*100003+index)
        patch=tuple(SETTINGS['patch'])
        assert all(n>=p for n,p in zip(gt.shape,patch))
        if rng.rand()<.7 and (gt>0).any():
            chosen=np.argwhere(gt>0)[rng.randint(int((gt>0).sum()))]
            origin=tuple(int(np.clip(c-p//2,0,n-p)) for c,p,n in zip(chosen,patch,gt.shape))
        else:
            origin=tuple(int(rng.randint(0,n-p+1)) for n,p in zip(gt.shape,patch))
        spans=tuple(slice(a,a+p) for a,p in zip(origin,patch))
        target=gt[spans].copy()
        noisy,events=corrupt(target,gt,origin,rng)
        prob=probability_from_mask(noisy,rng)
        coords=np.stack(np.broadcast_arrays(*[
            (2*np.arange(a,a+p,dtype=np.float32)/max(n-1,1)-1).reshape(tuple(p if j==i else 1 for j in range(3)))
            for i,(a,p,n) in enumerate(zip(origin,patch,gt.shape))]))
        assert prob.shape==(9,*patch) and np.isfinite(prob).all() and np.allclose(prob.sum(0),1,atol=1e-5)
        assert target.min()>=0 and target.max()<=8
        return dict(prob=torch.from_numpy(prob),image=torch.from_numpy(image[spans][None].copy()),
            coord=torch.from_numpy(coords.copy()),target=torch.from_numpy(target.astype(np.int64)),
            patient=patient,events='|'.join(events),changed=int(np.count_nonzero(noisy!=target)))
