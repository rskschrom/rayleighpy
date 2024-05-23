import numpy as np
from scipy.special import elliprd

# polarizability for ellipsoid
def ellipsoid(a, b, c, eps):
    '''
    Get the polarizabilities for an ellipsoid.
    
    Parameters
    ----------
    a : float
        The a axis length.
    b : float
        The b axis length.
    c : float
        The c axis length.
    eps : float complex
        The complex refractive index of the ellipsoid.
    
    Returns
    -------
    alpha_a : float complex
        The polarizability along the a axis of the ellipsoid.
    alpha_b : float complex
        The polarizability along the b axis of the ellipsoid.
    alpha_c : float complex
        The polarizability along the c axis of the ellipsoid.
    '''
    # calculate shape factors
    lc = a*b*c/3.*elliprd(a**2., b**2., c**2.)
    lb = a*b*c/3.*elliprd(a**2., c**2., b**2.)
    la = 1.-lb-lc
    
    # get polarizabilities
    alpha_a = 4.*np.pi*a*b*c*(eps-1.)/(3.+3.*la*(eps-1.))
    alpha_b = 4.*np.pi*a*b*c*(eps-1.)/(3.+3.*lb*(eps-1.))
    alpha_c = 4.*np.pi*a*b*c*(eps-1.)/(3.+3.*lc*(eps-1.))
    
    return alpha_a, alpha_b, alpha_c