import numpy as np
from rayleighpy import transform
from rayleighpy import polarizability
from rayleighpy import fields

# test principle component tensor rotation
def test_pc_rotate():
    tens_tr = np.squeeze(transform.pc_rotate(3.,2.,1., np.pi/12.,-5./8.*np.pi,7./12.*np.pi))
    tens_test = np.array([[1.96664025+0.j,-0.27882943+0.j,0.04621684+0.j],
                          [-0.27882943+0.j,1.38343021+0.j,-0.71991552+0.j],
                          [0.04621684+0.j,-0.71991552+0.j,2.64992955+0.j]]
)
    assert np.sqrt(np.mean(np.abs(tens_tr-tens_test)**2.))<1.e-8

# test conversion from bh to hv convention
def test_bh_hv_basis():
    # get incident and scattering bases
    phi = np.linspace(0., 360., 73)*np.pi/180.
    theta = np.linspace(0., 180., 37)*np.pi/180.
    phif, thetaf = np.meshgrid(phi, theta, indexing='ij')
    phif = phif.flatten()
    thetaf = thetaf.flatten()

    # set particle orientation
    alpha = 0.
    beta = 20.*np.pi/180.
    gamma = 0.

    # create polarizability tensor
    alp_a, alp_b, alp_c = polarizability.ellipsoid(2.,0.9,0.1, 3.17)
    alp_tens = transform.pc_rotate(alp_a, alp_b, alp_c, alpha, beta, gamma)

    # get scattering amplitude matrices in bohren and huffman convention and convert to hv
    smat_bh = fields.ampl_scat_mat_bh(phi, theta, alp_tens, 2.*np.pi/32.1)
    smat_bh.shape = (2,2,1,len(phi)*len(theta))
    smat_hv = transform.bh_hv_basis(smat_bh, phif)[:,:,0,:]

    # get scattering amplitude matrices in spherical coor convention
    smat = fields.ampl_scat_mat(0., 0., phif, thetaf, alp_tens, 2.*np.pi/32.1)
    smat = np.squeeze(smat)

    # compare hv and sph conventions
    err = np.sqrt(np.mean(np.abs(smat_hv-smat)**2.))
    assert err<1.e-16
