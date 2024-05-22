from rayleighpy import polarizability
import numpy as np

# test ellipsoid
def test_ellipsoid():
    alp_a, alp_b, alp_c = polarizability.ellipsoid(1.,1.,1.,2.)
    assert (alp_a,alp_b,alp_c)==(np.pi,np.pi,np.pi)
