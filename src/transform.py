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
    alpha : float, ndarray
        The first Euler angle rotation (zyz convention, radians).
    beta : float, ndarray
        The second Euler angle rotation (zyz convention, radians).
    gamma : float, ndarray
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
    rot = Rotation.from_euler('ZYZ', np.array([alpha,beta,gamma]).T)
    rmat = rot.as_matrix()
    
    # add dimension if rotation angles are scalar
    if len(rmat.shape)==2:
        rmat.shape = (1,3,3)
        
    tensor_tr = np.einsum('ij,ljk->lik', tensor, rmat)
    tensor_tr = np.einsum('lij,lik->jkl', rmat, tensor_tr)
    return tensor_tr

def tensor_scat(tensor, basis_inc, basis_sca, fscat=False, bh=False):
    '''
    Transform the polarizability tensor into the 2x2 far-field scattering basis given the incident and scattering polarization bases.
    
    Parameters
    ----------
    tensor : ndarray
        The (3,3,L) polarizability tensor.
    basis_inc : ndarray
        A (3,3,N) array of basis vectors for the incident electric field with the columns of the array containing the basis vectors `e_v`, `e_h`, and `e_r` in that order.
    basis_sca : ndarray
        A (3,3,M) array of basis vectors for the scattered electric field with the columns of the array containing the basis vectors `e_v`, `e_h`, and `e_r` in that order.
    fscat : bool
        Whether or not to do the calculation for forward scattering, where the scattered directions equal the incident directions.
    bh : bool
        Whether to use the Bohren and Huffman (1983) scattering plane convention.
        
    Returns
    -------
    tensor_sca : ndarray
        The (2,2,L,N,M) tensor in the incident and scattered polarization bases. In the case where `fscat=True`, the output dimension of the tensor is (2,2,L,N).
    '''
    vh_inc = basis_inc[:,:2,:]
    vh_sca = basis_sca[:,:2,:]
    
    # expand tensor dimensions if there is only a single tensor
    if len(tensor.shape)==2:
        tensor.shape = (3,3,1)
    
    if fscat:
        tensor_sca = np.einsum('ijo,jkp->ikop', tensor, vh_inc)
        tensor_sca = np.einsum('ilp,ikop->lkop', vh_sca, tensor_sca)
    elif bh:
        tensor_sca = np.einsum('ijo,jkp->ikop', tensor, vh_inc)
        tensor_sca = np.einsum('ilpq,ikop->lkopq', vh_sca, tensor_sca)
    else:
        tensor_sca = np.einsum('ijo,jkp->ikop', tensor, vh_inc)
        tensor_sca = np.einsum('ilq,ikop->lkopq', vh_sca, tensor_sca)
    
    return tensor_sca

def bh_hv_basis(smat_bh, phi):
    '''
    Get the scattering amplitude matrices in the h-v basis from scattering amplitude matrices in the Bohren and Huffman (1983) basis.
    
    Parameters
    ----------
    smat_bh : ndarray
        The (2,2,N,M) array of scattering ampltiude matrices for `N` particle orientations and `M` scattering directions in the Bohren and Huffman convention for incident direction along the z axis.
    phi : ndarray
        The 1D array of length `M` of scattering plane phi angles.
        
    Returns
    -------
    smat_hv : ndarray
        The (2,2,N,M) array of scattering amplitude matrices in the h-v convention.
    '''
    # 2x2 rotation matrix
    rmat = np.array([[np.cos(phi),np.sin(phi)],
                     [-np.sin(phi),np.cos(phi)]])
    smat_hv = np.einsum('ijkl,jpl->ipkl', smat_bh, rmat)
    return smat_hv

def scatdir_lab(smat_par, phi_sca, theta_sca):
    '''
    Get the scattering amplitude matrices in lab-frame directions for scattering amplitude matrices in the particle frame (i.e., incident propagation direction along the z-axis).
    
    Parameters
    ----------
    smat_par : ndarray
        The (2,2,N,M) array of scattering ampltiude matrices for `N` particle orientations and `M` scattering directions in the h-v convention for incident direction along the z axis.
    phi_sca : ndarray
        The 1D array of length `M` of scattering phi angles relative to the z-axis propagation direction.
    theta_sca : ndarray
        The 1D array of length `M` of scattering theta angles relative to the z-axis propagation direction.
        
    Returns
    -------
    smat_lab : ndarray
        The (2,2,N,M) array of scattering amplitude matrices in the lab frame.
    '''
    # 2x2 rotation matrix
    rmat = np.array([[np.cos(phi),-np.sin(phi)],
                     [np.sin(phi),np.cos(phi)]])
    #smat_hv = np.einsum('ijk,jlpk->ilpk', rmat, smat_bh)
    smat_hv = np.einsum('ijkl,jpl->ipkl', smat_bh, rmat)
    return smat_hv