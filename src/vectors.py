import numpy as np

def spherical_basis(phi, theta):
    '''
    Get the basis vectors corresponding to spherical coordinates angles `phi` and `theta`. The basis vectors are `e_v`, `e_h`, and `e_r`, where `e_v x e_h = e_r`.

    Parameters
    ----------
    phi : float, ndarray
        The azimuthal angle from the x axis.
    theta : float, ndarray
        The zenith angle from the z axis.

    Returns
    -------
    basis : ndarray
        The basis vectors return in columns of the (3,3) ndarray in the order of `e_v`, `e_h`, and `e_r`. 
    '''
    e_v = np.array([np.cos(theta)*np.cos(phi),
                    np.cos(theta)*np.sin(phi),
                    -np.sin(theta)])
    e_h = np.array([-np.sin(phi),
                    np.cos(phi),
                    phi-phi])
    e_r = np.array([np.sin(theta)*np.cos(phi),
                    np.sin(theta)*np.sin(phi),
                    np.cos(theta)])
    
    # expand dimensions for vectors in the case of scalar phi and theta
    if len(e_v.shape)==1:
        e_v.shape = (3,1)
        e_h.shape = (3,1)
        e_r.shape = (3,1)
        
    basis = np.concatenate((e_v[:,np.newaxis,:],
                            e_h[:,np.newaxis,:],
                            e_r[:,np.newaxis,:]), axis=1)
    return basis
