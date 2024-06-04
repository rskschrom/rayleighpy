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
    
def ampl_scat_mat_bh(phi_1d_sca, theta_1d_sca, alp_tens, k):
    '''
    Get the amplitude scattering matrices for a polarizability tensor for incident direction along the z axis and a set of scattering angles using the Bohren and Huffman (1983) convention.
    
    Parameters
    ----------
    phi_1d_sca : ndarray (N,)
        The scattering plane angles (phi) in radians.
    theta_1d_sca : ndarray (M,)
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
    bas_inc = spherical_basis(phi_1d_sca, np.zeros(len(phi_1d_sca)))
    
    # 2d grid for theta and phi scattered
    nphi = len(phi_1d_sca)
    nthet = len(theta_1d_sca)
    phi2d, theta2d = np.meshgrid(phi_1d_sca, theta_1d_sca, indexing='ij')
    phi_sca = phi2d.flatten()
    theta_sca = theta2d.flatten()
    
    bas_sca = spherical_basis(phi_sca, theta_sca)
    bas_sca.shape = (3,3,nphi,nthet)
    smat = 1j*k**3./(4.*np.pi)*tensor_scat(alp_tens, bas_inc, bas_sca, bh=True)
    
    return smat
