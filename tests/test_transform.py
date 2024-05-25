import numpy as np
from rayleighpy import transform

# test principle component tensor rotation
def test_pc_rotate():
    tens_tr = np.squeeze(transform.pc_rotate(3.,2.,1., np.pi/12.,-5./8.*np.pi,7./12.*np.pi))
    tens_test = np.array([[1.96664025+0.j,-0.27882943+0.j,0.04621684+0.j],
                          [-0.27882943+0.j,1.38343021+0.j,-0.71991552+0.j],
                          [0.04621684+0.j,-0.71991552+0.j,2.64992955+0.j]]
)
    assert np.sqrt(np.mean(np.abs(tens_tr-tens_test)**2.))<1.e-8
