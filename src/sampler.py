import numpy as np
import torch

from nsd_gnet8x.src.file_utility import zip_dict
from nsd_manifold.src.geometry import gs

class Subsampler(object):
    def __init__(self, random_subspace=None, random_projections=None, orthogonalize=False):
        super(Subsampler).__init__()
        
        self.rs = random_subspace
        self.rp = random_projections
        self.sample_index = None
        self.proj_mat = None
        self.orth = orthogonalize
        
    def apply(self, fmaps):
        '''
            fmaps is a dictionary of feature spaces
        '''
        sfm = {}     
        if self.rs is not None: # apply the random selection
            if self.sample_index is None: # create the random sampler list
                self.sample_index = {}
                for k,fm in fmaps.items():
                    ffm = fm.reshape((len(fm), -1))
                    ps = ffm.shape[1]
                    idx = np.arange(ps)
                    np.random.shuffle(idx)
                    tf = min(self.rs, ps)
                    self.sample_index[k] = idx[:tf]
        
            for k,fm,sm in zip_dict(fmaps, self.sample_index):
                sfm[k] = fm.reshape((len(fm), -1))[:,sm]
        else:
            for k,fm in fmaps.items():
                sfm[k] = fm.reshape((len(fm), -1))
            
        if self.rp is not None: # apply only the random projection  
            if self.proj_mat is None: # create the random projections matrices list
                self.proj_mat = {}
                for k,fm in sfm.items():
                    ps = fm.shape[1]
                    self.proj_mat[k] = np.random.randn(ps, self.rp) / np.sqrt(self.rp)
                    if self.orth and ps<=self.rp:
                        self.proj_mat[k] = gs(self.proj_mat[k].T).T
           
            for k in sfm.keys(): # inplace projection
                sfm[k] = sfm[k]@self.proj_mat[k]
 
        return sfm

    
def get_value(_x):
    return np.copy(_x.data.cpu().numpy())
def set_value(_x, x):
    _x.data.copy_(torch.from_numpy(x))

def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 

def extract_image_features(image_loader, sampler):
    sfeats, pred_id = [], []
    for images, labels in image_loader:
        pred_id += [labels.numpy(),] 
        sfeats += [sampler.apply([images]),]   
    pred_id = np.concatenate(pred_id, axis=0)
    sfeats  = [np.concatenate(a) for a in zip(*sfeats)]
    return pred_id, sfeats

def extract_feature_maps_batch(_fmaps_fn, images, batchsize=100, device='cuda:0'):
    feature_maps = []
    # create buffer on first iteration
    for k,_fm in enumerate(_fmaps_fn(images[:batchsize].to(device))):       
        feature_maps += [np.zeros(shape=(len(images),)+tuple(_fm.size()[1:]), dtype=np.float32),]
        feature_maps[-1][:batchsize] = get_value(_fm)
    # loop over images
    for rr, rl in tqdm(iterate_range(batchsize, len(images)-batchsize, batchsize)):
        for k,_fm in enumerate(_fmaps_fn(images[rr].to(device))):
            feature_maps[k][rr] = get_value(_fm)
    return feature_maps

def extract_sampled_features(_fmaps_fn, image_loader, sampler, batchsize=100, device='cuda:0'):
    sfeats, pred_id = [], []
    for images, labels in image_loader:
        pred_id += [labels.numpy(),] 
        sfeats += [sampler.apply(extract_feature_maps_batch(_fmaps_fn, images, batchsize=batchsize, device=device)),]   
    pred_id = np.concatenate(pred_id, axis=0)
    sfeats  = [np.concatenate(a) for a in zip(*sfeats)]
    return pred_id, sfeats