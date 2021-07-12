#import illustris_python.groupcat as gc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

def load_box(data_path, BoxID=0, fixed_cosmo=True):
    if fixed_cosmo:
        filename = os.path.join(data_path, 'abacus','pairs_planck_00_box_{:02d}.hdf5'.format(BoxID))
    else:
        filename = '../data/pairs_box_{:02d}.hdf5'.format(BoxID)
    data = {}
    f = h5py.File(filename, 'r')
    for k in f.keys():
        data[k] = f[k][...]
    f.close()
    print("Finished reading {}".format(filename))
    
    dtype=[('BoxID','i8'), ('hubble', 'f8'), ('omega_de', 'f8'),
      ('omega_m', 'f8'), ('n_s', 'f8'), ('sigma_8', 'f8'), ('w_0', 'f8')]
    cosmo_data = np.loadtxt(os.path.join(data_path, "abacus", "box_cosmo_params.dat"), dtype=dtype)
    hubble = cosmo_data['hubble'][BoxID]
    
    # compute center of mass velocity
    a = data['vel_A'].copy()
    b = data['vel_B'].copy()
    v_cm = data['vel_A'].copy()
    mass_tot = data['mass_A'] + data['mass_B']
    for i in range(3):
        a[:,i] = data['vel_A'][:,i] * data['mass_A']/mass_tot
        b[:,i] = data['vel_B'][:,i] * data['mass_B']/mass_tot
        v_cm[:,i] = a[:,i] + b[:,i]
    data['vel_CM'] = v_cm
    data['vel_CM_mag'] = np.sqrt(np.sum(v_cm**2, axis=1))
    
    
    
    data['vel_A_mag'] = np.sqrt(np.sum(data['vel_A']**2, axis=1))
    data['vel_B_mag'] = np.sqrt(np.sum(data['vel_B']**2, axis=1))
    data['vel_G_mag'] = np.sqrt(np.sum(data['vel_G']**2, axis=1))

    data['pos_AB'] = np.sqrt(np.sum( (data['pos_B'] - data['pos_A'])**2, axis=1))
    data['vel_AB'] = np.sqrt(np.sum( (data['vel_B'] - data['vel_A'])**2, axis=1)) # comoving
    data['vel_AB_rad'] = np.sum((data['pos_B'] - data['pos_A'])*(data['vel_B'] - data['vel_A']), axis=1)/data['pos_AB'] #comoving
    data['vel_AB_tan'] = np.sqrt((data['vel_AB']**2 - data['vel_AB_rad']**2))# comoving
    
    #data['vel_CM'] = data['vel_B']
    
    #now we compute the radial velocity including the hubble flow
    data['vel_AB_rad'] = data['vel_AB_rad'] + (data['pos_AB'] * hubble)

    # here we compute the dot product between the position vector and the radial velocity
    data['mu'] = np.sum((data['pos_B']-data['pos_A'])*(data['vel_B']), axis=1)/(data['vel_B_mag']*data['pos_AB'])
    data['mu_vv'] = np.sum(data['vel_A']*data['vel_B'], axis=1)/(data['vel_A_mag']*data['vel_B_mag'])

    # here we compute the dot product between the position vector and the center of mass velocity
    data['mu_cm'] = np.sum((data['pos_B']-data['pos_A'])*(data['vel_CM']), axis=1)/(data['vel_CM_mag']*data['pos_AB'])
    
    datos = {}
    ii = (data['pos_A'][:,0] > 10) & (data['pos_A'][:,0]<710)
    keys = ['vel_A_mag', 'vel_B_mag', 'pos_AB', 'vel_AB', 'vel_AB_rad', 'vel_AB_tan', 'vmax_A', 'vmax_B', 'mu', 'mu_vv', 'mu_cm', 'vel_CM_mag']
    for kk in keys:
        datos[kk] = data[kk][ii]
    keys = ['vel_G_mag', 'vmax_G']
    for kk in keys:
        datos[kk] = data[kk][:]
    return datos
        

def count_pairs_FOF_abacus_box(BoxID=0):
    pair_count = {}
    data = load_box(BoxID=BoxID)
    print("processing box ", BoxID)
    vlim = {'mean':627, 'sigma':22}

    # Selection in vmax and kinematics
    ii = (data['vmax_A']<240) & (data['vmax_B']<240) 
    ii &= (data['vel_AB_rad']<0) 
    ii &= (np.abs(data['vel_AB_rad'])>np.abs(data['vel_AB_tan']))
    
    ll = (data['vmax_G']<240)
    
    # Selection of high velocity
    jj = (data['vel_B_mag'] > vlim['mean'])
    mm = (data['vel_G_mag'] > vlim['mean'])
    
    # Count
    pair_count['pair_total'] = np.count_nonzero(ii)
    pair_count['pair_high'] = np.count_nonzero(ii&jj)
    pair_count['individual_total'] = np.count_nonzero(ll)
    pair_count['individual_high'] = np.count_nonzero(ll&mm)
    pair_count['mean_mu_total'] = np.median(data['mu'][ii])
    pair_count['mean_mu_high'] = np.median(data['mu'][ii&jj])
    pair_count['std_mu_total'] = np.std(data['mu'][ii])
    pair_count['std_mu_high'] = np.std(data['mu'][ii&jj])
    return pair_count

def all_data():
    outfile = "../data/summary_pair_count.dat"
    f = open(outfile, "w")
    for i in range(40):
        pair_count = count_pairs_FOF_abacus_box(BoxID=i)
        f.write("{:d} {:d} {:d} {:d} {:d} {:f} {:f} {:f} {:f}\n".format(i, 
                                                   pair_count['individual_total'], 
                                                   pair_count['individual_high'],
                                                   pair_count['pair_total'],
                                                   pair_count['pair_high'],
                                                   pair_count['mean_mu_total'],
                                                   pair_count['mean_mu_high'],
                                                   pair_count['std_mu_total'],
                                                   pair_count['std_mu_high']))
    f.close()
    
