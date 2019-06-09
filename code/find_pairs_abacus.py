import illustris_python.groupcat as gc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import abacus_cosmos.Halos as ach
from astropy.table import Table
import h5py

def compute_pairs_FOF_abacus_box(BoxID=0):
    basePath = "/Users/forero/github/abacus/data/AbacusCosmos_720box_{:02d}_FoF_halos_z0.100/".format(BoxID)
    
    print("Started reading the data")
    halo_data = ach.read_halos_FoF(basePath)
    print("Finished reading the data")
    
    BoxSize = 720.0
    halo_data['pos'] = halo_data['pos']+BoxSize/2.0

    print("Vcirc selection")
    ii = halo_data['vcirc_max']>200 # in units of km/s
    S_pos = halo_data['pos'][ii]
    S_vel = halo_data['vel'][ii]
    S_vmax = halo_data['vcirc_max'][ii]
    S_parent_fof = halo_data['id'][ii]
    n_S = len(S_pos)
    print("Number of halos selected:", n_S)
    
    print("Started Neighbor computation")
    nbrs_S = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(S_pos)
    dist_S, ind_S = nbrs_S.kneighbors(S_pos)
    print("Finished Neighbor computation")
    
    print("Started isolation computation")
    neighbor_index = ind_S[:,1]
    neighbor_list = ind_S[:,2:]

    n_pairs = 0

    halo_A_id = np.empty((0), dtype=int)
    halo_B_id = np.empty((0), dtype=int)

    for i in range(n_S):
        l = neighbor_index[neighbor_index[i]]% n_S
        j = neighbor_index[i] % n_S
    
        other_j = neighbor_list[i,:] % n_S
        other_l = neighbor_list[neighbor_index[i],:] % n_S
    
        if((i==l) & (not (j in halo_A_id)) & (not (i in halo_B_id))): # first check to find mutual neighbors
            vmax_i = S_vmax[i]
            vmax_j = S_vmax[j]
            vmax_limit = min([vmax_i, vmax_j])
                
            pair_d = dist_S[i,1] # This is the current pair distance
            dist_limit = pair_d * 3.0 # exclusion radius for massive structures
            
            massive_close_to_i = any((dist_S[i,2:]<dist_limit) & (S_vmax[other_j] >= vmax_limit))
            massive_close_to_j = any((dist_S[j,2:]<dist_limit) & (S_vmax[other_l] >= vmax_limit))
            if((not massive_close_to_i) & (not massive_close_to_j)): # check on massive structures inside exclusion radius
                n_pairs = n_pairs+ 1
                halo_A_id = np.append(halo_A_id, int(i))
                halo_B_id = np.append(halo_B_id, int(j))
    print("Finished isolation computation")
    print("Pairs found:", n_pairs)
    
    filename = '../data/pairs_box_{:02d}.hdf5'.format(BoxID)
    print("Started writing data to ", filename)

    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('pos_A', data=S_pos[halo_A_id,:])
    h5f.create_dataset('pos_B', data=S_pos[halo_B_id,:])
    h5f.create_dataset('pos_G', data=S_pos)
    h5f.create_dataset('vel_A', data=S_vel[halo_A_id,:])
    h5f.create_dataset('vel_B', data=S_vel[halo_B_id,:])
    h5f.create_dataset('vel_G', data=S_vel)
    h5f.create_dataset('vmax_A', data=S_vmax[halo_A_id])
    h5f.create_dataset('vmax_B', data=S_vmax[halo_B_id])
    h5f.create_dataset('vmax_G', data=S_vmax)
    h5f.close()
    return 

for i in range(2):
    compute_pairs_FOF_abacus_box(BoxID=i)
    
