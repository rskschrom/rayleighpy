import numpy as np
from scipy.spatial.transform import Rotation

def pc_rotate(pa, pb, pc, alpha, beta, gamma):
    '''
    Get the transformed tensor from the three principle components and the Euler rotation angles.

    Parameters
    ----------
    pa : float complex
        The value of along principle axis a.
    pb : float complex
        The value of along principle axis b.
    pc : float complex
        The value of along principle axis c.
    alpha : float
        The first Euler angle rotation (zyz convention, radians).
    beta : float
        The second Euler angle rotation (zyz convention, radians).
    gamma : float
        The third Euler angle rotation (zyz convention, radians).
    
    Returns
    -------
    tensor_tr : ndarray
        The transformed tensor for the principle components.
    '''
    tensor = np.zeros([3,3], dtype=complex)
    tensor[0,0] = pa
    tensor[1,1] = pb
    tensor[2,2] = pc
    rot = Rotation.from_euler('ZYZ', [alpha,beta,gamma])
    rmat = rot.as_matrix()
    print(rmat)
    tensor_tr = rmat.T @ (tensor @ rmat)
    return tensor_tr

def tensor_scat(tensor, basis_inc, basis_sca):
    '''
    Transform the polarizability tensor into the 2x2 far-field scattering basis given the incident and scattering polarization bases.
    
    Parameters
    ----------
    tensor : ndarray
        The (3,3) polarizability tensor.
    basis_inc : ndarray
        A (3,3,N) array of basis vectors for the incident electric field with the columns of the array containing the basis vectors `e_v`, `e_h`, and `e_r` in that order.
    basis_sca : ndarray
        A (3,3,M) array of basis vectors for the scattered electric field with the columns of the array containing the basis vectors `e_v`, `e_h`, and `e_r` in that order.
    
    Returns
    -------
    tensor_sca : ndarray
        The (2,2,N,M) tensor in the incident and scattered polarization bases.
    '''
    vh_inc = basis_inc[:,:2,:]
    vh_sca = basis_sca[:,:2,:]
    tensor_sca = np.einsum('ij,jkp->ikp', tensor, vh_inc)
    tensor_sca = np.einsum('ilq,ikp->lkpq', vh_sca, vh_inc)
    return tensor_sca
