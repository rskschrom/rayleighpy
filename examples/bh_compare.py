from rayleighpy import polarizability
from rayleighpy import transform
from rayleighpy import vectors
from rayleighpy import fields
import numpy as np
import matplotlib.pyplot as plt
from convert_rotations import smat_inc_transform

# get incident and scattering bases

phi = np.linspace(0., 360., 73)*np.pi/180.
theta = np.linspace(0., 180., 37)*np.pi/180.
phi_inc, theta_inc = np.meshgrid(phi, theta, indexing='ij')
phi_inc = phi_inc.flatten()
theta_inc = theta_inc.flatten()

oi = 542
alpha = 0.
beta = theta_inc[oi]
gamma = phi_inc[oi]

# create polarizability tensor
alp_a, alp_b, alp_c = polarizability.ellipsoid(2.,2.,0.1, 3.17)
print(alp_a, alp_b, alp_c)

'''
# randomly sample orientations
nang = 2000
alpha = np.random.rand(nang)*2.*np.pi
beta = np.random.rand(nang)*np.pi
gamma = np.zeros([nang])
#gamma = np.random.rand(nang)*2.*np.pi
'''
alp_tens = transform.pc_rotate(alp_a, alp_b, alp_c, phi_inc-phi_inc, theta_inc, phi_inc)
print(alp_tens.shape)
norient = len(phi_inc)

# single orientation polarizability tensor
alp_tens_single = transform.pc_rotate(alp_a, alp_b, alp_c, alpha, beta, gamma)
print(alp_tens_single.shape)

'''
bas_inc = vectors.spherical_basis(phi_inc, theta_inc)
bas_sca = vectors.spherical_basis(phi_sca, theta_sca)
print(bas_inc)
print(bas_sca)
'''
#alp_sca = transform.tensor_scat(alp_tens, bas_inc, bas_sca)
# get scattering amplitude matrices in bohren and huffman convention
wavl = 32.1
k = 2.*np.pi/wavl
smat_bh = fields.ampl_scat_mat_bh(phi, theta, alp_tens_single, k)
#smat_bh.shape = (2,2,norient,len(phi)*len(theta))
smat_bh.shape = (2,2,1,norient)


#oi = 0
#smat_tr = smat_inc_transform(smat_bh[:,:,oi,:], 0., beta*180./np.pi, gamma*180./np.pi, 0., 0., theta_inc*180./np.pi, phi_inc*180./np.pi)
#print(smat_tr.shape)

# get hv basis
smat_hv = transform.bh_hv_basis(smat_bh, phi_inc)

# get scattering amplitude matrices in spherical coor convention
smat = fields.ampl_scat_mat(phi_inc, theta_inc, phi_inc, theta_inc, alp_tens_single, k)
smat = np.squeeze(smat)

# get forward scattering amplitudes
smat_fs = fields.ampl_fscat_mat(phi_inc, theta_inc, alp_tens_single, k)
smat_fs = np.squeeze(smat_fs)
print(smat.shape, smat_bh.shape, smat_fs.shape)

# test plot
#zi = np.argmin(alpha**2.+beta**2.)
#print(alpha[zi], beta[zi])

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(2,1,1)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat[0,0,0,:])**2, s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,1,2)
#plt.scatter(phi_inc, theta_inc, c=np.abs(smat_tr[0,0,:])**2, s=10., cmap='Spectral_r')
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_hv[0,0,0,:])**2, s=10., cmap='Spectral_r')

plt.colorbar()
plt.savefig('scat.png')
