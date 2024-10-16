import numpy as np
import matplotlib.pyplot as plt
from src.utility import invalid_filter, offdiagonal


def highlight_mask(img, cat, lw=1):
    cim = img==cat
    shift_right = np.concatenate([cim[1:], cim[-1, np.newaxis]], axis=0)
    shift_up    = np.concatenate([cim[:,1:], cim[:, -1, np.newaxis]], axis=1)
    mask = np.logical_or(cim!=shift_right, cim!=shift_up)
    i = 1
    while i<lw:
        shift_right = np.concatenate([mask[1:], mask[-1, np.newaxis]], axis=0)
        shift_up    = np.concatenate([mask[:,1:], mask[:, -1, np.newaxis]], axis=1)
        mask = np.logical_or(mask, np.logical_or(shift_right, shift_up))
        i += 1
    return mask 

def other_mask(img, cat):
    return img!=cat

def apply_highlight(img, hl, color=[1,1,1]):
    im = np.copy(img)
    im = im.reshape((-1, img.shape[2]))
    im[hl.flatten()] = np.array(color)
    return im.reshape(img.shape) 

def apply_mask(img, mask, color=[1,1,1], alpha=.5):
    im = np.copy(img)
    im = im.reshape((-1, img.shape[2]))
    im[mask.flatten()] = im.reshape((-1, img.shape[2]))[mask.flatten()]*(1-alpha)  + np.array(color)*alpha
    return im.reshape(img.shape)

class Colorizer(object):
    def __init__(self, cmap='gray', scale='linear', vmin=0, vmax=1):
        from matplotlib.colors import Normalize, LogNorm
        self.normer = None
        if scale=='log':
            self.normer = LogNorm(vmin=vmin, vmax=vmax, clip=True)
        else:
            self.normer = Normalize(vmin=vmin, vmax=vmax, clip=True)
        lcolors = cm.ScalarMappable(cmap=cmap)
        lcolors.set_array([])  
        self.cmapper = lcolors.get_cmap() 
        
    def __call__(self, x):
        return self.cmapper(self.normer(x))

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    from matplotlib import transforms
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def plot_SNR_summary(manifold_SNR, sample_cmap, plot_roi, subj_violins=False, perc_cutoff=5, position_offset=0, fs=16, legend=False):
    from src.utility import invalid_filter
    
    for k,name in enumerate(['SNR', 'Sign', 'Bias', 'Nsa', 'Nsb', 'Nss', 'Ds']):
        plt.subplot(4,2,k+1)
        Ymin, Ymax = np.inf, -np.inf
        subj_all, subj_means, subj_stds = [], [], []
        for roi in plot_roi:
            Y = []
            for s,d in manifold_SNR.items():
                y = invalid_filter(d[name][roi])
                Ymin, Ymax = min(np.percentile(y, perc_cutoff), Ymin), max( np.percentile(y, 100-perc_cutoff), Ymax)
                Y += [np.mean(y),]
            Y = np.array(Y)
            subj_all   += [Y,]
            subj_means += [np.mean(Y, axis=0),]
            subj_stds  += [np.std(Y, axis=0),]
        subj_all  = np.array(subj_all)
        subj_means = np.array(subj_means)
        subj_stds  = np.array(subj_stds)

        for i,s in enumerate(manifold_SNR.keys()):
            plt.plot(np.arange(len(plot_roi)) + position_offset, subj_all[:,i], marker='None', lw=2, color=sample_cmap[s], label='S%01d'%s)   
        plt.plot(np.arange(len(plot_roi)) + position_offset, subj_means, marker='o', ms=10, lw=5, color='k')   
        plt.plot(np.arange(len(plot_roi)) + position_offset, subj_means, marker='o', ms=6, lw=2, color='w')   
        if k>=5:
            plt.gca().set_xticks(np.arange(len(plot_roi)) + position_offset)
            plt.gca().set_xticklabels(plot_roi, rotation=90)
        else:
            plt.gca().set_xticks(np.arange(len(plot_roi)))
            plt.gca().set_xticklabels([])
        plt.ylim([Ymin, Ymax])

        props = dict(boxstyle='round', facecolor='white', alpha=1)
        plt.gca().text(0.6, 0.9, 'fillertext', transform=plt.gca().transAxes, fontsize=16, color='w',
            verticalalignment='top', bbox=props)
        plt.gca().text(0.6, 0.9, name, transform=plt.gca().transAxes, fontsize=16, color='k',
            verticalalignment='top', bbox=props)
        #plt.ylabel(name)
        if k==6 and legend:
            leg=plt.legend(loc=(1.01, 0.0))
            
            
            

def brain5shot_sample_average(err_dict, sample_cmap, roi_order, percentile_cutoff=5, return_bounds=True):
    '''
        sample_names should be the relevant keys of err_dict
    '''
    from src.utility import offdiagonal
    sample_names = sample_cmap.keys()
    subj_avg = []    
    gYmin, gYmax = np.inf, -np.inf
    for s in sample_names:

        Y = np.array([offdiagonal(err_dict[s]['err'][r]) for r in roi_order])      
        Ymin, Ymax = [np.percentile(y, percentile_cutoff) for y in Y], [np.percentile(y, 100-percentile_cutoff) for y in Y]
        Yf = [y[np.logical_and(y>ymin, y<ymax)] for y,ymin,ymax in zip(Y, Ymin, Ymax)]
        gYmin = min(Ymin) if min(Ymin)<gYmin else gYmin
        gYmax = max(Ymax) if max(Ymax)>gYmax else gYmax
        
        parts = plt.violinplot(Yf, \
            positions=np.arange(len(Y))+.12*s-0.54, \
            vert=True, widths=0.1, showmeans=False, showextrema=False, showmedians=False)

        for pc in parts['bodies']:
            pc.set_facecolor(sample_cmap[s])
            pc.set_edgecolor(sample_cmap[s])
            pc.set_alpha(1)

        values = np.array([np.mean(y) for y in Y])
        _=plt.plot(np.arange(len(Y))+.12*s-0.54, values, marker='o', ms=6, color='k', linestyle='None')
        subj_avg += [values,]
    if return_bounds:
        return np.array(subj_avg), gYmin, gYmax
    else:
        return np.array(subj_avg) 

    
    
def geom_sample_average(geom_dict, geom_name, sample_names, sample_color, plot_roi, \
                        mask=None, percentile_cutoff=5, return_bounds=True, pos_lims=None, pos_offset=0, width_mult=1., plot_violins=True, aggregate_samples=False):
    '''
    Plot the mean of the geometric characteristic geom_name for a series of geometries, as well as the optional distributions for each samples/subject, shifted for visibility.
    '''
    if len(sample_names)==1 or aggregate_samples:
        dw = .8*width_mult
        wi = 0
    else:
        dw = .8*width_mult / (len(sample_names)-1)
        wi = -.4*width_mult   
    subj_avg, Yfs = [], []
    gYmin, gYmax = np.inf, -np.inf
    for j,s in enumerate(sample_names):
        if mask is not None:
            Y = np.array([invalid_to_num((geom_dict[s][geom_name][r].flatten())[mask], np.mean) for r in plot_roi])
        else:
            Y = np.array([offdiagonal(geom_dict[s][geom_name][r]) for r in plot_roi])
        Ymin, Ymax = np.array([np.percentile(y, percentile_cutoff) for y in Y]), np.array([np.percentile(y, 100-percentile_cutoff) for y in Y])
        Yf = [y[np.logical_and(y>ymin, y<ymax)] for y,ymin,ymax in zip(Y, Ymin, Ymax)]
        #print ('Yf: ', [y.shape for y in Yf])    # [ROI list for points]
        Yfs += [Yf,]      
        values = np.array([np.mean(y) for y in Y])
        gYmin = min(Ymin) if min(Ymin)<gYmin else gYmin
        gYmax = max(Ymax) if max(Ymax)>gYmax else gYmax
        subj_avg += [values,]
        
        if not aggregate_samples:
            Xcenter = np.arange(len(Y))
            if pos_lims is not None:
                Xcenter = np.linspace(pos_lims[0], pos_lims[1], len(Y))
                dL = pos_lims[1] - pos_lims[0]
                dW, Wi = dw / (dL*len(Y)), wi / (dL*len(Y))
            else:
                dW, Wi = dw, wi 
            Xcenter = Xcenter + pos_offset    
            X = Xcenter + Wi + dW*j
            if plot_violins:
                parts = plt.violinplot(Yf, positions=X, vert=True, widths=dW*.8, showmeans=False, showextrema=False, showmedians=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(sample_color[s])
                    pc.set_edgecolor(sample_color[s])
                    pc.set_alpha(1)
            _=plt.plot(X, [np.mean(y) for y in Yf], marker='o', ms=6, color='k', linestyle='None')
    
    if aggregate_samples:
        Xcenter = np.arange(len(Y))
        if pos_lims is not None:
            Xcenter = np.linspace(pos_lims[0], pos_lims[1], len(Y))
            dL = pos_lims[1] - pos_lims[0]
            dW, Wi = dw / (dL*len(Y)), wi / (dL*len(Y))
        else:
            dW, Wi = dw, wi
        Xcenter = Xcenter + pos_offset  
        Yf = [np.concatenate([Yfs[s][k] for s in range(len(Yfs))]) for k in range(len(Yfs[0]))]
        #print ('Yf: ', [y.shape for y in Yf])
        if plot_violins:
            parts = plt.violinplot(Yf, positions=Xcenter, vert=True, widths=dW*.5, showmeans=False, showextrema=False, showmedians=False)
            for pc in parts['bodies']:
                pc.set_facecolor('k')
                pc.set_edgecolor('k')
                pc.set_alpha(.5)
        _=plt.plot(Xcenter, [np.mean(y) for y in Yf], marker='o', ms=6, color='k', linestyle='None') 
    if return_bounds:
        return Xcenter, np.array(subj_avg), gYmin, gYmax
    else:
        return Xcenter, np.array(subj_avg)
    

def braingeom_sample_average(geom_dict, geom_name, sample_cmap, roi_order, mask=None, percentile_cutoff=5, return_bounds=True):
    sample_names = sample_cmap.keys()
    subj_avg = []    
    gYmin, gYmax = np.inf, -np.inf
    for j,s in enumerate(sample_names):
        
        ndims = len(geom_dict[s][geom_name][roi_order[0]].shape)
        if mask is not None:
            if ndims==2:
                Y = np.array([invalid_to_num((geom_dict[s][geom_name][r].flatten())[mask], np.mean) for r in roi_order])
            elif ndims==3:
                Y = np.array([invalid_to_num((np.mean(geom_dict[s][geom_name][r], axis=0).flatten())[mask], np.mean) for r in roi_order])
            else:
                print ('  >> unimplemented')
                return
        else:
            if ndims==2:
                Y = np.array([offdiagonal(geom_dict[s][geom_name][r]) for r in roi_order])
            elif ndims==3:
                Y = np.array([offdiagonal(np.mean(geom_dict[s][geom_name][r], axis=0)) for r in roi_order])
            else:
                print ('  >> unimplemented')
                return
            
        Ymin, Ymax = [np.percentile(y, percentile_cutoff) for y in Y], [np.percentile(y, 100-percentile_cutoff) for y in Y]
        Yf = [y[np.logical_and(y>ymin, y<ymax)] for y,ymin,ymax in zip(Y, Ymin, Ymax)]
        gYmin = min(Ymin) if min(Ymin)<gYmin else gYmin
        gYmax = max(Ymax) if max(Ymax)>gYmax else gYmax
        
        parts = plt.violinplot(Yf, \
            positions=np.arange(len(Y))+.12*(j+1)-0.54, \
            vert=True, widths=0.1, showmeans=False, showextrema=False, showmedians=False)

        for pc in parts['bodies']:
            pc.set_facecolor(sample_cmap[s])
            pc.set_edgecolor(sample_cmap[s])
            pc.set_alpha(1)

        values = np.array([np.mean(y) for y in Y])
        _=plt.plot(np.arange(len(Y))+.12*(j+1)-0.54, values, marker='o', ms=6, color='k', linestyle='None')
        subj_avg += [values,]
    if return_bounds:
        return np.array(subj_avg), gYmin, gYmax
    else:
        return np.array(subj_avg) 