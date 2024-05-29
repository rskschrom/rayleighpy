from rayleighpy import polarizability
from rayleighpy import transform
from rayleighpy import vectors
from rayleighpy import fields
import numpy as np
import matplotlib.pyplot as plt

# get incident and scattering bases

phi = np.linspace(0., 360., 73)*np.pi/180.
theta = np.linspace(0., 180., 37)*np.pi/180.
phi_inc, theta_inc = np.meshgrid(phi, theta, indexing='ij')
phi_inc = phi_inc.flatten()
theta_inc = theta_inc.flatten()


# create polarizability tensor
alp_a, alp_b, alp_c = polarizability.ellipsoid(2.,0.7,0.1, 3.17)
print(alp_a, alp_b, alp_c)

'''
# randomly sample orientations
nang = 2000
alpha = np.random.rand(nang)*2.*np.pi
beta = np.random.rand(nang)*np.pi
gamma = np.zeros([nang])
#gamma = np.random.rand(nang)*2.*np.pi
'''
alp_tens = transform.pc_rotate(alp_a, alp_b, alp_c, 0., 0., 0.)
print(alp_tens.shape)


'''
bas_inc = vectors.spherical_basis(phi_inc, theta_inc)
bas_sca = vectors.spherical_basis(phi_sca, theta_sca)
print(bas_inc)
print(bas_sca)
'''
#alp_sca = transform.tensor_scat(alp_tens, bas_inc, bas_sca)
# get scattering amplitude matrices
wavl = 32.1
k = 2.*np.pi/wavl
smat = fields.ampl_scat_mat_bh(phi, theta, alp_tens, k)
#smat = fields.ampl_fscat_mat(phi_inc, theta_inc, alp_tens, k)
print(smat.shape)
smat.shape = (2,2,1,len(phi)*len(theta))

# test plot
#zi = np.argmin(alpha**2.+beta**2.)
#print(alpha[zi], beta[zi])

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(2,1,1)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat[0,0,0,:])**2, s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,1,2)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat[1,1,0,:])**2, s=10., cmap='Spectral_r')
plt.colorbar()
plt.savefig('scat.png')
