import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.stats as scst
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sim', help='simulation ID', required=True)
args = parser.parse_args()


BoxID = int(args.sim)
filename = '../data/abacus/pairs_planck_00_box_{:02d}.hdf5'.format(BoxID)
data = {}
f = h5py.File(filename, 'r')
for k in f.keys():
    data[k] = f[k][...]
f.close()
print('finished reading {}'.format(filename))



dtype=[('BoxID','i8'), ('hubble', 'f8'), ('omega_de', 'f8'),
      ('omega_m', 'f8'), ('n_s', 'f8'), ('sigma_8', 'f8'), ('w_0', 'f8')]
cosmo_data = np.loadtxt("../data/abacus/box_cosmo_params.dat", dtype=dtype)
hubble = cosmo_data['hubble'][BoxID]
print('hubble parameter', hubble)

a = data['vel_A'].copy()
b = data['vel_B'].copy()
v_cm = data['vel_A'].copy()
mass_tot = data['mass_A'] + data['mass_B']
for i in range(3):
    a[:,i] = data['vel_A'][:,i] * data['mass_A']/mass_tot
    b[:,i] = data['vel_B'][:,i] * data['mass_B']/mass_tot
    v_cm[:,i] = a[:,i] + b[:,i]
v_cm_norm = np.sqrt(np.sum(v_cm**2, axis=1))
data['vel_A_mag'] = np.sqrt(np.sum(data['vel_A']**2, axis=1))

data['vel_A_mag'] = np.sqrt(np.sum(data['vel_A']**2, axis=1))
data['vel_B_mag'] = np.sqrt(np.sum(data['vel_B']**2, axis=1))
data['vel_G_mag'] = np.sqrt(np.sum(data['vel_G']**2, axis=1))

data['pos_AB'] = np.sqrt(np.sum( (data['pos_B'] - data['pos_A'])**2, axis=1))
data['vel_AB'] = np.sqrt(np.sum( (data['vel_B'] - data['vel_A'])**2, axis=1))
data['vel_AB_rad'] = np.sum((data['pos_B'] - data['pos_A'])*(data['vel_B'] - data['vel_A']), axis=1)/data['pos_AB']
data['vel_AB_tan'] = np.sqrt((data['vel_AB']**2 - data['vel_AB_rad']**2))


#now we compute the radial velocity including the hubble flow
data['vel_AB_rad'] = data['vel_AB_rad'] + (data['pos_AB'] * hubble)

datos = {}
ii = (data['pos_A'][:,0] > 0) & (data['pos_A'][:,0]<720)
keys = ['vel_A_mag', 'vel_B_mag', 'pos_AB', 'vel_AB', 'vel_AB_rad', 'vel_AB_tan', 'vmax_A', 'vmax_B', 'mass_A', 'mass_B']
v_cm_norm = v_cm_norm[ii]
for kk in keys:
#    print(kk)
    datos[kk] = data[kk][ii]
keys = ['vel_G_mag', 'vmax_G']
for kk in keys:
   # print(kk)
    datos[kk] = data[kk][:]
    
ii = (datos['vel_AB_rad']<0) & (datos['pos_AB']<1.0) & ((datos['mass_A']*1E10 < 5E12) & (datos['vmax_B']*1E10 <5E12)) 

cm_vel = v_cm_norm[ii]
tan_vel = datos['vel_AB_tan'][ii]
rad_vel = datos['vel_AB_rad'][ii]
tot_mass = datos['mass_A'][ii]+ datos['mass_B'][ii]

results  = np.array([cm_vel, tan_vel, rad_vel, tot_mass])

print(np.shape(results))



# write positions
fileout = '../data/abacus/summary_velocities_abacus_planck_00_box_{:02d}.dat'.format(BoxID)
np.savetxt(fileout, results.T, fmt='%f %f %f %f')
print(' wrote results data to {}'.format(fileout))




