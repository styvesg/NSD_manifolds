import numpy as np

def round10up(x):
    return 10**np.ceil(np.log10(x))
def round10down(x):
    return 10**np.floor(np.log10(x))

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def get_n_random_thing_samples(things_mh, subj, a_thing, n):
    things = things_mh[subj][:,a_thing]
    thing_sample_indices = np.arange(len(things))[things.astype(bool)]    
    if n>len(thing_sample_indices):
        print ('there are less than %d things from this sort, (%d)'%(n, len(thing_sample_indices)))
        return thing_sample_indices
    np.random.shuffle(thing_sample_indices)
    return thing_sample_indices[:n]

def nanfunc(v, func=lambda x: x):
    print ('SWITCH TO invalid_filter')
    return func((v.flatten())[np.logical_and(~np.isnan(v.flatten()), ~np.isinf(v.flatten()))])

def invalid_filter(v):
    '''
    Flatten and remove all invalid entry (NaN or inf) from an array.
    '''
    return (v.flatten())[np.logical_and(~np.isnan(v.flatten()), ~np.isinf(v.flatten()))]

def offdiagfunc(v, func=lambda x: x):
    print ('SWITCH TO offdiagonal')
    return func((v.flatten())[~np.eye(len(v)).astype(bool).flatten()])

def offdiagonal(v):
    '''
    Return a flattened view of the off diagonal elements of an array. For an array of size (N, N), the returned vector will be of length N**2 - N.
    '''
    return (v.flatten())[~np.eye(len(v)).astype(bool).flatten()]

def invalid_to_num(x, value_fn=lambda x:x*0):
    '''
    Similar to invalid_filter() but instead replaces the invalid entries (NaN or inf) to a given function of the remaining values. This is done in order to preserve the length of the array
    '''
    sh = x.shape
    z = x.flatten()
    m = np.logical_and(~np.isnan(x.flatten()), ~np.isinf(x.flatten()))
    z[~m] = value_fn(z[m])
    return z


# hierarchical distance can't quite explain the pattern of 5-shot error because HD is symmetric while 5-shot err isn't
def corrmat(a, b, mask=None):
    m = ~np.eye(len(a)).astype(bool).flatten()
    if mask is not None:
        m = np.logical(m, mask)
    return np.corrcoef((a.flatten())[m], (b.flatten())[m])[0,1]

def symmetry_score(a):
    '''
    Correlation score between lower and upper diagonal
    '''
    m = ~np.eye(len(a)).astype(bool).flatten()
    return np.corrcoef(a.T.flatten()[m], a.flatten()[m])[0,1]

def symmetric_antisymmetric_mat(a):
    return (a+a.T)/2, (a-a.T)/2

def set_zero_diag(a):
    print ('SWITCH TO set_diagonal_value')
    mdiag = np.eye(len(a)).astype(bool).flatten()
    A = a.flatten()
    A[mdiag] = 0
    return A.reshape(a.shape)

def set_diagonal_value(a, fill_value=0):
    mdiag = np.eye(len(a)).astype(bool).flatten()
    A = a.flatten()
    A[mdiag] = fill_value
    return A.reshape(a.shape)


def dir_proj(x, y, slope):  
    v = np.array([[1, slope], [-slope, 1]])
    v /= np.sqrt(np.sum(v**2, axis=1))
    
    U = np.stack([x.flatten(), y.flatten()], axis=1)
    return (U@v[0]).reshape(x.shape), (U@v[1]).reshape(y.shape)