import numpy as np
from .vectors import spherical_basis
from .transform import tensor_scat

def ampl_scat_mat(phi_inc, theta_inc, phi_sca, theta_sca, alp_tens, k):
    '''
    Get the amplitude scattering matrices for a polarizability tensor for a single incident angle and a set of scattering angles.
    
    Parameters
    ----------
    phi_inc : ndarray (N,)
        The incident phi angles in radians.
    theta_inc : ndarray (N,)
        The incident theta angles in radians.
    phi_sca : ndarray (M,)
        The scattered phi angles in radians.
    theta_sca : ndarray (M,)
        The scattered theta angles in radians.
    alp_tens : ndarray
       A (3,3,L) complex array representing the polarizability tensor.
    k : float
       The wave number for the incident wave.
    Returns
    -------
    smat : ndarray
      A (3,3,L,N,M) array of scattering amplitude matrices for each of the `N` scattering angle pairs.
    '''
    # get basis vectors
    bas_inc = spherical_basis(phi_inc, theta_inc)
    bas_sca = spherical_basis(phi_sca, theta_sca)
    smat = 1j*k**3./(4.*np.pi)*tensor_scat(alp_tens, bas_inc, bas_sca)
    
    return smat

def ampl_fscat_mat(phi, theta, alp_tens, k):
    '''
    Get the amplitude scattering matrices in the forward scattering direction for a polarizability tensor for a set of angles.
    
    Parameters
    ----------
    phi : float
        The incident (and scattering) phi angle in radians.
    theta : float
        The incident (and scattering) theta angle in radians.
    alp_tens : ndarray
       A (3,3) complex array representing the polarizability tensor.
    k : float
       The wave number for the incident wave.
    Returns
    -------
    smat : ndarray
      A (3,3,L,N) array of scattering amplitude matrices for each of the `N` scattering angle pairs.
    '''
    # get basis vectors
    bas_inc = spherical_basis(phi, theta)
    smat = 1j*k**3./(4.*np.pi)*tensor_scat(alp_tens, bas_inc, bas_inc, fscat=True)
    
    return smat