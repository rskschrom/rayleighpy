import numpy as np

def spherical_basis(phi, theta):
    '''
    Get the basis vectors corresponding to spherical coordinates angles `phi` and `theta`. The basis vectors are `e_v`, `e_h`, and `e_r`, where `e_v x e_h = e_r`.

    Parameters
    ----------
    phi : float
        The azimuthal angle from the x axis.
    theta : float
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
                    0.])
    e_r = np.array([np.sin(theta)*np.cos(phi),
                    np.sin(theta)*np.sin(phi),
                    np.cos(theta)])
    basis = np.vstack((e_v,e_h,e_r)).T
    return basis

print(spherical_basis(22.*np.pi/180.,63.*np.pi/180.))
