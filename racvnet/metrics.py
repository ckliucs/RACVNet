"""Read-only 3D mask diagnostics in original D,H,W voxel coordinates."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

SURFACE = ndi.generate_binary_structure(3,1)


def boundary(mask):
    return mask ^ ndi.binary_erosion(mask,structure=SURFACE,iterations=1,border_value=0)


def surfaces(mask):
    return np.argwhere(boundary(mask)).astype(np.float64)


def distance_arrays(p,g):
    ps,gs=surfaces(p),surfaces(g)
    if not len(ps) or not len(gs):
        return ps,gs,np.empty(0),np.empty(0)
    p2g=cKDTree(gs).query(ps,k=1,eps=0,workers=1)[0]
    g2p=cKDTree(ps).query(gs,k=1,eps=0,workers=1)[0]
    return ps,gs,p2g,g2p


def metrics(p,g):
    p,g=p.astype(bool),g.astype(bool)
    ps,gs,dp,dg=distance_arrays(p,g)
    pn,gn=int(p.sum()),int(g.sum())
    tp=int(np.count_nonzero(p&g))
    hd=float(np.percentile(np.concatenate((dp,dg)),95)) if pn and gn else 0.
    dice=2.*tp/(pn+gn) if pn+gn else 1.
    return dict(dice=dice,hd95_voxel=hd,tp=tp,fp=pn-tp,fn=gn-tp,
                prediction_voxels=pn,gt_voxels=gn,prediction_present=int(pn>0),gt_present=int(gn>0),
                empty_hd95_bookkeeping=int(not(pn and gn))), (ps,gs,dp,dg)
