from tqdm import tqdm
import numpy as np
from scipy.special import erf
from sklearn.decomposition import PCA

def nanfunc(v, func=lambda x: x):
    return func((v.flatten())[np.logical_and(~np.isnan(v.flatten()), ~np.isinf(v.flatten()))])

def offdiagfunc(v, func=lambda x: x):
    return func((v.flatten())[~np.eye(len(v)).astype(bool).flatten()])

def k_rand_label_index(lid, k, i):
    idx = np.arange(len(lid))[lid==i]
    np.random.shuffle(idx)
    return idx[:k], idx[k:]

def invalid_to_num(x, value_fn=lambda x:x*0):
    sh = x.shape
    z = x.flatten()
    m = np.logical_and(~np.isnan(x.flatten()), ~np.isinf(x.flatten()))
    z[~m] = value_fn(z[m])
    return z

def gs(A):
    B = np.copy(A)
    B[0,:] /= np.sqrt(np.sum(B[0,:]**2))
    for k in range(1, len(B)):
        B[k,:] = B[k,:] - np.dot(np.dot(B[k,:], B[:k].T), B[:k])
        B[k,:] /= np.sqrt(np.sum(B[k,:]**2))
    return B

def PCA_geometry(R):
    pca  = PCA()
    try:
        R_pca = pca.fit_transform(R)
    except:
        print ('-------------------------------')
        print ('could not calculate PCs')
        print ('sample shape = ', R.shape)
        print ('isnan = ', np.sum(np.isnan(R)).astype(bool))
        print ('isinf = ', np.sum(np.isinf(R)).astype(bool))
        print ('-------------------------------')
        
        return np.mean(R, axis=0), np.ones(R.shape[1]), np.eye(R.shape[1])

    R_pca = pca.fit_transform(R)
    r = np.sqrt(pca.explained_variance_ * len(R)) # radii
    #D = np.sum(r**2)**2/ np.sum(r**4) # participation ratio
    #r_avg = np.sqrt(np.mean(r**2)) # average radius
    U = pca.components_ #* r[:, np.newaxis] / r_avg# 
    r0 = np.mean(R, axis=0) # centroid
    #return r0, D, r, r_avg, U
    return r0, r, U

def prD(r):
    return np.sum(r**2,axis=-1)**2 / np.sum(r**4, axis=-1)

def H(z):
    return .5 * (1 - erf(z / np.sqrt(2))) * 100

def randNsphere(P, D):
    ang = np.random.uniform(0, 2*np.pi, size=(P, D-1))
    Z = []
    for i in range(0, D):
        z = np.ones(shape=(P,))
        for j in range(0, D-1-i):
            z *= np.sin(ang[:,j])
        if i>0:
            for j in range(max(D-i-1,0),max(D-i,0)):
                z *= np.cos(ang[:,j])
        Z += [z,]
    return np.array(Z).T

def compute_SNR(signal, bias, D, nsa, nsb, nno, m):
    return 1/2*(signal + bias/m)/ np.sqrt(1/D[:,None]/m + nsa + (nsb + nno)/m )

def geometry(centers, Rs, Us, m, P):
    '''Adapted from Sorscher et al 2021, "the geometry of concept learning"
    https://github.com/bsorsch/geometry_fewshot_learning'''
    K = len(centers) # number of concepts
    dists = np.sqrt(((centers[:,None] - centers[None])**2).sum(-1))
    dist_norm = dists / np.sqrt((Rs**2).sum(-1)[:,None] / P)
    Ds = prD(Rs)
    # Center-subspace
    csa, csb, nno = [], [], []
    for a in tqdm(range(K)):
        for b in range(K):
            if a!=b:
                dx0 = centers[a] - centers[b]
                dx0hat = dx0 / np.linalg.norm(dx0)
                costheta_a = Us[a]@dx0hat
                csa.append((costheta_a**2 * Rs[a]**2).sum() / (Rs[a]**2).sum())
                costheta_b = Us[b]@dx0hat
                csb.append((costheta_b**2 * Rs[b]**2).sum() / (Rs[a]**2).sum())
                cosphi = Us[a]@Us[b].T
                ss_overlap = (cosphi**2*Rs[a][:,None]**2*Rs[b]**2).sum() / (Rs[a]**2).sum()**2
                nno.append(ss_overlap)
            else:
                csa.append(np.nan)
                csb.append(np.nan)
                nno.append(np.nan)
    csa, csb, nno = np.stack(csa).reshape(K,K), np.stack(csb).reshape(K,K), np.stack(nno).reshape(K,K)             
    signal = dist_norm**2
    nsa = csa * signal
    nsb = csb * signal
    bias = (Rs**2).sum(-1) / (Rs**2).sum(-1)[:,None] - 1
    SNR = compute_SNR(signal, bias, Ds, nsa, nsb, nno, m)
    return signal, bias, Ds, nsa, nsb, nno, SNR


#################
# deprecated

def compute_errors(manifolds, m, n_samples):
    '''Adapted from Sorscher et al 2021, "the geometry of concept learning"
    https://github.com/bsorsch/geometry_fewshot_learning'''
    err_all = np.zeros((len(manifolds),len(manifolds)))
    err_std = np.zeros((len(manifolds),len(manifolds)))
    for a in tqdm(range(len(manifolds))):
        Xa = manifolds[a]
        for b in range(len(manifolds)):
            Xb = manifolds[b]
            errs = []
            for _ in range(n_samples):
                perma = np.random.permutation(len(Xa))
                permb = np.random.permutation(len(Xb))

                xa,ya = np.split(Xa[perma],(m,))
                xb,yb = np.split(Xb[permb],(m,))
                w = (xa-xb).mean(0)
                mu = (xa+xb).mean(0)/2

                h = ya@w - w@mu
                err = (h<0).mean()
                errs.append(err)
            err_all[a,b] = np.mean(errs)
            err_std[a,b] = np.std(errs)
    np.fill_diagonal(err_all,np.nan)   
    return err_all, err_std

def compute_geometries(manifolds, m):
    r0s, Rs, Us = [],[],[]
    for manifold in tqdm(manifolds):
        r0a, Ra, Ua = PCA_geometry(manifold)
        r0s += [r0a,]
        Rs  += [Ra]
        Us  += [Ua,]
    P = manifolds.shape[1] # manifolds shape: [K, P, V]
    return geometry(np.array(r0s), np.array(Rs), np.array(Us), m, P)



#################################################### 
### Multisubject/Multi-feature(ROI) loops
####################################################
## expand dictionary with derived measures
def expand_geometry(geom, subjects=None):
    if subjects is None:
        subjects = geom.keys()
    for s in subjects:
        m = geom['m'] if 'm' in geom else geom[s]['m']  
        geom[s]['N'], geom[s]['DR'], geom[s]['DL'], geom[s]['DRinv'], geom[s]['DLinv'] = {}, {}, {}, {}, {}
        for r in geom[s]['SNR'].keys():
            geom[s]['N'][r] = geom[s]['Nsa'][r] + \
                (geom[s]['Nsb'][r] + geom[s]['Nss'][r]) / m 
            geom[s]['DR'][r] = np.outer(geom[s]['Ds'][r], np.ones_like(geom[s]['Ds'][r]))
            geom[s]['DL'][r] = np.outer(np.ones_like(geom[s]['Ds'][r]), geom[s]['Ds'][r])
            geom[s]['DRinv'][r] = geom[s]['DR'][r]**(-1)
            geom[s]['DLinv'][r] = geom[s]['DL'][r]**(-1)
            
            
from src.sampler import *
def feature_subsampling_and_split(feature_dicts, labels, random_subspace=None, random_projections=300, err_frac=.1):
    '''
        feature_dicts [#subjects][#ROI] --> array[#samples, #dims]
        labels (array or dict [#subjects] of array) --> array[#samples]{int} or array[#sample, #labels]{bool}
    '''
    err_feature_embs, geom_feature_embs, err_labels, geom_labels = {}, {}, {}, {}
    min_err_samples, min_geom_samples = [], []
    for s,dd in feature_dicts.items():
        if type(labels)==dict: # if labels is a dict, subjectwise labels. Otherwise, shared labels 
            lab = labels[s]
        else:
            lab = labels
        
        sampler = Subsampler(random_subspace=random_subspace, random_projections=random_projections)
        emb = sampler.apply(dd)

        sidx = np.arange(len(lab))
        np.random.shuffle(sidx)

        err_select = sidx[:int(err_frac*len(lab))]
        geom_select = sidx[int(err_frac*len(lab)):]

        err_feature_embs[s]  = {r: e[err_select] for r,e in emb.items()}
        err_labels[s]  = lab[err_select]
        geom_feature_embs[s] = {r: e[geom_select] for r,e in emb.items()}
        geom_labels[s] = lab[geom_select]

        min_err_samples  += [int(np.min(np.sum(err_labels[s], axis=0))),]
        min_geom_samples += [int(np.min(np.sum(geom_labels[s], axis=0))),]
    
    return (err_feature_embs, err_labels, min(min_err_samples)), (geom_feature_embs, geom_labels, min(min_geom_samples))
    #print ('fold', f, ', P (err)=', P)
    
    
def calculate_manifolds_error(feature_dicts, labels, m, n_samples=1):
    '''
    feature_dicts [#subject][#ROI]
    '''
    manifold_err = {}
    for s,dd in feature_dicts.items():
        if type(labels)==dict: # if labels is a dict, subjectwise labels. Otherwise, shared labels 
            lab = labels[s]
        else:
            lab = labels            
        if len(lab.shape)==1:
            #print (' >> single-labels')
            K = len(sorted(np.unique(lab)))
        elif len(lab.shape)==2:
            #print (' >> multi-labels')
            K = lab.shape[1]
        else:
            print (' >> unimplemented')
            return
        ###
        manifold_err[s] = {'err': {}, 'std': {}, 'm': m, 'n_samples': n_samples}
        for r,fm in dd.items():
            ###
            K_shot_err1 = np.full(shape=(K, K), fill_value=0, dtype=np.float32)  
            K_shot_err2 = np.full(shape=(K, K), fill_value=0, dtype=np.float32)
            for l1 in tqdm(range(K-1)):
                if len(lab.shape)==1:
                    cidxa = np.arange(len(lab))[lab==l1]    
                elif len(lab.shape)==2:
                    cidxa = np.arange(len(lab))[lab[:,l1].astype(bool)]
                ###
                for l2 in range(l1+1, K):
                    if len(lab.shape)==1:
                        cidxb = np.arange(len(lab))[lab==l2]    
                    elif len(lab.shape)==2:
                        cidxb = np.arange(len(lab))[lab[:,l2].astype(bool)]

                    err_ab, err_ba = [], []
                    for _ in range(n_samples):
                        np.random.shuffle(cidxa)
                        np.random.shuffle(cidxb)

                        ma = np.mean(fm[cidxa[:m]], axis=0, keepdims=True) # [1, voxels]
                        mb = np.mean(fm[cidxb[:m]], axis=0, keepdims=True) # [1, voxels] 
                        # how many of the remaining obj1 example are closer to m1 than m2?
                        x = fm[cidxa[m:]]
                        K_shot_err1[l1,l2] += np.mean((np.sum((x-ma)**2, axis=1) > np.sum((x-mb)**2, axis=1)).astype(np.float)) / n_samples
                        K_shot_err2[l1,l2] += np.mean((np.sum((x-ma)**2, axis=1) > np.sum((x-mb)**2, axis=1)).astype(np.float))**2 / n_samples
                        # how many of the remaining obj2 example are closer to m2 than m1?
                        x = fm[cidxb[m:]]
                        K_shot_err1[l2,l1] += np.mean((np.sum((x-ma)**2, axis=1) < np.sum((x-mb)**2, axis=1)).astype(np.float)) / n_samples
                        K_shot_err2[l2,l1] += np.mean((np.sum((x-ma)**2, axis=1) < np.sum((x-mb)**2, axis=1)).astype(np.float))**2 / n_samples
            ###
            manifold_err[s]['err'][r] = K_shot_err1
            manifold_err[s]['std'][r] = np.sqrt(K_shot_err2 -  K_shot_err1**2)
    return manifold_err


def calculate_manifolds_directions(feature_dicts, labels, P):
    '''
        feature_dicts: dict[ #sample/#subject ][ #layer/#ROI ] --> dirs[#sample][prop][#ROI]
    '''
    manifold_data = {}
    for s,dd in feature_dicts.items():
        lab = labels
        if type(labels)==dict: # if labels is a dict, subjectwise labels. Otherwise, shared labels     
            lab = labels[s]
        if len(lab.shape)==1:
            #print (' >> single-labels')
            K = len(sorted(np.unique(lab)))
        elif len(lab.shape)==2:
            #print (' >> multi-labels')
            K = lab.shape[1]
        else:
            print (' >> unimplemented')
            return      
        ###
        manifold_data[s] = {'R0': {}, 'Rs': {}, 'Us': {}, 'V': {}, 'P': P}
        for r,fm in dd.items():      
            V = fm.shape[1]
            R0s, Rs, Us = [],[],[]
            for l in tqdm(range(K)):
                if len(lab.shape)==1:
                    cidxa = np.arange(len(lab))[lab==l]    
                elif len(lab.shape)==2:
                    cidxa = np.arange(len(lab))[lab[:,l].astype(bool)]
                
                r0a, Ra, Ua = PCA_geometry(fm[cidxa[:P]])
                if r0a is not None:
                    R0s += [r0a,]
                    Rs  += [Ra]
                    Us  += [Ua,]
                else:
                    R0s += [np.zeros(shape=(V)),]
                    Rs += [np.zeros(shape=(V)),]
                    Us  += [np.zeros(shape=(V, V)),]            
            manifold_data[s]['V'][r] = V
            manifold_data[s]['R0'][r] = np.array(R0s)
            manifold_data[s]['Rs'][r] = np.array(Rs)
            manifold_data[s]['Us'][r] = np.array(Us)  
    return manifold_data

# save_stuff( output_dir + 'imagenet_alexnet_manifold_sample_%01d'%s, flatten_dict(manifold_data))
    
def calculate_manifolds_SNR(manifold_dirs, m=5):
    '''
        dirs[#sample][prop][#ROI] --> geom[#sample][prop][#ROI]
    '''
    manifold_dict = {}
    for s in manifold_dirs.keys():
        manifold_dict[s] = {'Sign': {}, 'Bias': {}, 'Nsa': {}, 'Nsb': {}, 'Nss': {}, 'Ds': {}, 'SNR': {}, 'm': m, 'P': manifold_dirs[s]['P']}
        for r in manifold_dirs[s]['R0'].keys():
            manifold_dict[s]['Sign'][r], manifold_dict[s]['Bias'][r], manifold_dict[s]['Ds'][r], \
            manifold_dict[s]['Nsa'][r], manifold_dict[s]['Nsb'][r], manifold_dict[s]['Nss'][r], \
            manifold_dict[s]['SNR'][r] = \
                geometry(manifold_dirs[s]['R0'][r], manifold_dirs[s]['Rs'][r], manifold_dirs[s]['Us'][r], m, manifold_dirs[s]['P'])
    return manifold_dict
# save_stuff( output_dir + 'imagenet_alexnet_manifold_sample_%01d_SNR'%s, flatten_dict(manifold_dict))
  