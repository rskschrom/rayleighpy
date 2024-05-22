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
        The first Euler angle rotation (zyz convention).
    beta : float
        The second Euler angle rotation (zyz convention).
    gamma : float
        The third Euler angle rotation (zyz convention).
    
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

