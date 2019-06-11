import illustris_python.groupcat as gc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import h5py

def load_box(BoxID=0):
    filename = '../data/pairs_box_{:02d}.hdf5'.format(BoxID)
    data = {}
    f = h5py.File(filename, 'r')
    for k in f.keys():
        data[k] = f[k][...]
    f.close()
    print("Finished reading {}".format(filename))
    
    dtype=[('BoxID','i8'), ('hubble', 'f8'), ('omega_de', 'f8'),
      ('omega_m', 'f8'), ('n_s', 'f8'), ('sigma_8', 'f8'), ('w_0', 'f8')]
    cosmo_data = np.loadtxt("../data/box_cosmo_params.dat", dtype=dtype)
    hubble = cosmo_data['hubble'][BoxID]
    
    data['vel_A_mag'] = np.sqrt(np.sum(data['vel_A']**2, axis=1))
    data['vel_B_mag'] = np.sqrt(np.sum(data['vel_B']**2, axis=1))
    data['vel_G_mag'] = np.sqrt(np.sum(data['vel_G']**2, axis=1))

    data['pos_AB'] = np.sqrt(np.sum( (data['pos_B'] - data['pos_A'])**2, axis=1))
    data['vel_AB'] = np.sqrt(np.sum( (data['vel_B'] - data['vel_A'])**2, axis=1)) # comoving
    data['vel_AB_rad'] = np.sum((data['pos_B'] - data['pos_A'])*(data['vel_B'] - data['vel_A']), axis=1)/data['pos_AB'] #comoving
    data['vel_AB_tan'] = np.sqrt((data['vel_AB']**2 - data['vel_AB_rad']**2))# comoving
    
    #now we compute the radial velocity including the hubble flow
    data['vel_AB_rad'] = data['vel_AB_rad'] + (data['pos_AB'] * hubble)


    datos = {}
    ii = (data['pos_A'][:,0] > 10) & (data['pos_A'][:,0]<710)
    keys = ['vel_A_mag', 'vel_B_mag', 'pos_AB', 'vel_AB', 'vel_AB_rad', 'vel_AB_tan', 'vmax_A', 'vmax_B']
    for kk in keys:
        datos[kk] = data[kk][ii]
    keys = ['vel_G_mag', 'vmax_G']
    for kk in keys:
        datos[kk] = data[kk][:]
    return datos
        

def compute_pairs_FOF_abacus_box(BoxID=0):
    pair_count = {}
    data = load_box(BoxID=BoxID)
    print("processing box ", BoxID)
    vlim = {'mean':627, 'sigma':22}

    # Selection in vmax
    ii = (data['vmax_A']<240) & (data['vmax_B']<240)
    ll = (data['vmax_G']<240)
    
    # Selection of high velocity
    jj = (data['vel_A_mag'] > vlim['mean'])
    mm = (data['vel_G_mag'] > vlim['mean'])
    
    # Count
    pair_count['pair_total'] = np.count_nonzero(ii)
    pair_count['individual_total'] = np.count_nonzero(ll)
    pair_count['pair_high'] = np.count_nonzero(ii&jj)
    pair_count['individual_high'] = np.count_nonzero(ll&mm)
    return pair_count

def all_data():
    outfile = "../data/summary_pair_count.dat"
    f = open(outfile, "w")
    for i in range(40):
        pair_count = compute_pairs_FOF_abacus_box(BoxID=i)
        f.write("{:d} {:d} {:d} {:d} {:d}\n".format(i, 
                                                   pair_count['individual_total'], 
                                                   pair_count['pair_total'],
                                                   pair_count['individual_high'],
                                                   pair_count['pair_high']))
    f.close()
    
