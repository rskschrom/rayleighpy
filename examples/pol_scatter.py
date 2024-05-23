from rayleighpy import polarizability
from rayleighpy import transform
from rayleighpy import vectors
from rayleighpy import fields
import numpy as np
import matplotlib.pyplot as plt

# create polarizability tensor
alp_a, alp_b, alp_c = polarizability.ellipsoid(2.,0.1,0.1, 3.17)
print(alp_a, alp_b, alp_c)

alp_tens = transform.pc_rotate(alp_a, alp_b, alp_c, 0., 0., 0.)
print(alp_tens)

# get incident and scattering bases
phi_inc = 0.
theta_inc = 0.

phi = np.linspace(0., 360., 73)*np.pi/180.
theta = np.linspace(0., 180., 37)*np.pi/180.
phi_sca, theta_sca = np.meshgrid(phi, theta, indexing='ij')
phi_sca = phi_sca.flatten()
theta_sca = theta_sca.flatten()

bas_inc = vectors.spherical_basis(phi_inc, theta_inc)
bas_sca = vectors.spherical_basis(phi_sca, theta_sca)
print(bas_inc)
print(bas_sca)

#alp_sca = transform.tensor_scat(alp_tens, bas_inc, bas_sca)
# get scattering amplitude matrices
wavl = 32.1
k = 2.*np.pi/wavl
smat = fields.ampl_scat_mat(phi_inc, theta_inc, phi_sca, theta_sca, alp_tens, k)
print(smat.shape)

# test plot
s_hh2 = np.abs(smat[1,1,0,:])**2.
plt.scatter(phi_sca, theta_sca, c=s_hh2, s=5., cmap='Spectral_r')
plt.colorbar()
plt.savefig('scat.png')