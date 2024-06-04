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

oi = 0
alpha = 0.
beta = 20.*np.pi/180.
gamma = 0.
print(alpha*180./np.pi, beta*180./np.pi, gamma*180./np.pi)

# create polarizability tensor
alp_a, alp_b, alp_c = polarizability.ellipsoid(2.,0.9,0.1, 3.17)
print(alp_a, alp_b, alp_c)

# single orientation polarizability tensor
alp_tens_single = transform.pc_rotate(alp_a, alp_b, alp_c, alpha, beta, gamma)
print(alp_tens_single[:,:,0])

# get scattering amplitude matrices in bohren and huffman convention
wavl = 32.1
k = 2.*np.pi/wavl
smat_bh = fields.ampl_scat_mat_bh(phi, theta, alp_tens_single, k)
print(smat_bh.shape)
smat_bh.shape = (2,2,1,len(phi)*len(theta))
smat_hv = transform.bh_hv_basis(smat_bh, phi_inc)[:,:,0,:]

# get scattering amplitude matrices in spherical coor convention
smat = fields.ampl_scat_mat(0., 0., phi_inc, theta_inc, alp_tens_single, k)
smat = np.squeeze(smat)
print(smat.shape)
smat_sing = smat[:,:,:]
print(smat_hv.shape, smat_sing.shape)

# plot
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(2,2,1)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_bh[0,0,0,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,2,2)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_bh[0,1,0:])**2., s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,2,3)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_bh[1,0,0,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,2,4)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_bh[1,1,0,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()
plt.savefig('scat_bh.png')
plt.close()

# fig2
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(2,2,1)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_sing[0,0,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,2,2)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_sing[0,1,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,2,3)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_sing[1,0,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,2,4)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_sing[1,1,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()
plt.savefig('scat_sph.png')
plt.close()

# fig3
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(2,2,1)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_hv[0,0,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,2,2)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_hv[0,1:])**2., s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,2,3)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_hv[1,0,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()

ax = fig.add_subplot(2,2,4)
plt.scatter(phi_inc, theta_inc, c=np.abs(smat_hv[1,1,:])**2., s=10., cmap='Spectral_r')
plt.colorbar()
plt.savefig('scat_hv.png')
plt.close()