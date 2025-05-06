import numpy as np
import torch
from tqdm import tqdm

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

    
def extract_fmaps_embedding(net, _fmaps_fn, stim_data, label_data, batchsize, n_samples, rand_sampling_dim, rand_proj_dim, zscore=True, shuffle_params=False, device='cuda:0'):
    '''
    # example usage:
    model_name = 'squeezeNet-rc'
feature_embs, labels = \
    extract_fmaps_embedding(net, _fmap_fn, stim_data, things_mh, batchsize=100, n_samples=9, rand_sampling_dim=20000, rand_proj_dim=2000, zscore=True, device=device)
    '''
    from nsd_manifold.src.geometry import k_rand_label_index, geometry, PCA_geometry, prD
    
    subjects = list(stim_data.keys())
    rs = subjects[np.random.randint(len(subjects))]
    _x = torch.tensor(stim_data[rs][:batchsize]).to(device) # the input variable.
    _fmaps = _fmaps_fn(_x) 
                            
    feature_embs, labels = {}, {}
    for s in tqdm(np.arange(n_samples)):
        ##
        if shuffle_params:
            for _p in net.parameters():
                p = get_value(_p).flatten()
                np.random.shuffle(p)
                set_value(_p, p.reshape(_p.size()))
            net.eval()
        ##
        keep_idxes = [] # random sampling outside the sampler to reduce memory usage for the extracted feature maps
        rs = subjects[np.random.randint(len(subjects))] # random set of images, if more than one
        for k,_fm in enumerate(_fmaps):
            nf =  np.prod(_fm.size()[1:])
            idx = np.arange(nf)
            np.random.shuffle(idx)
            if nf<rand_sampling_dim:
                keep_idxes += [idx,]
                if s==0:
                    print (_fm.size()[1:], '-->', nf)
            else:
                keep_idxes += [idx[:rand_sampling_dim],]  
                if s==0:
                    print (_fm.size()[1:], '-->', nf, '-->', rand_sampling_dim)
        ##
        feature_maps = {}
        for k,(idx, _fm) in enumerate(zip(keep_idxes, _fmaps)):       
            feature_maps[k] = np.zeros(shape=(len(stim_data[rs]), len(idx)), dtype=np.float32)
        for rr, rl in iterate_range(0, len(stim_data[rs]), batchsize):
            _x = torch.tensor(stim_data[rs][rr]).to(device) # the input variable.
            _fmaps = _fmaps_fn(_x)
            for k,(idx,_fm) in enumerate(zip(keep_idxes, _fmaps)):
                _fm = torch.flatten(_fm, start_dim=1)[:,idx]
                feature_maps[k][rr] = get_value(_fm)
        ##  zscore      
        for k,fm in feature_maps.items():
            ffm = fm.reshape((fm.shape[0], -1))
            if zscore:
                feature_maps[k] = ffm - np.mean(ffm, axis=0, keepdims=True)
                feature_maps[k] /= np.std(ffm, axis=0, keepdims=True) + 1e-6
            else:
                feature_maps[k] = ffm
        ##
        sampler = Subsampler(random_projections=rand_proj_dim)
        feature_embs[s] = sampler.apply(feature_maps)    
        labels[s] = label_data[rs]
    return feature_embs, labels 
    