#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:15:15 2019

@author: smrak
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#fn = '/media/smrak/gnss/hdf/2013_0215T0000-0216T0000_conus046.yaml_20el_60s.h5'
fn = 'G:\\My Drive\\scintillation_data\\tid\\20150417\\2015_0417T0000-0418T0000_all0417.yaml_30el_30s_roti.h5'
#fn = 'G:\\My Drive\\scintillation_data\\tid\\20150301\\2015_0301T0000-0302T0000_all0301.yaml_30el_30s_roti.h5'
#fn = 'G:\\My Drive\\scintillation_data\\tid\\20150317\\2015_0317T0000-0318T0000_all0317.yaml_30el_30s_roti.h5'
#fn = 'G:\\My Drive\\scintillation_data\\tid\\20150417\\2015_0417T0000-0418T0000_reduced0417_d1_r2.yaml_30el_1s.h5'
D = h5py.File(fn, 'r')
po = D['po'][:]
po_length = D['po_length'][:]
N = D['rx_name'].size
D.close()

po_length[po_length >= 8*60] = np.nan

fig = plt.figure(figsize=[8, 5])
plt.title('Pool size: {}'.format(po.size))
plt.hist(list(po), range=(1,21), bins=20, density=True, color='b', align='left', rwidth=0.9)
plt.xlabel('Polynomial order')
plt.ylabel('Probability')
plt.xticks(np.arange(0,21.1,3))
#plt.xlim([3,polynom_list[-1]])
#plt.savefig(saveroot + 'po_hist.png', dpi=100)
#plt.close(fig)

fig = plt.figure(figsize=[8, 5])
plt.title('Pool size: {}'.format(po.size))
plt.hist(list(po_length), bins=50, density=True, color='b', align='left', rwidth=0.9)
plt.xlabel('LOS length [min]')
plt.ylabel('Probability')
#plt.savefig(saveroot + 'polength_hist.png', dpi=100)
#plt.close(fig)

h, x0, y0 = np.histogram2d(po, po_length, bins=[19,40], range=[[1,20], [0,500]])
fig = plt.figure(figsize=[8, 5])
plt.title('Total number: {}, Rx={}'.format(po.size, N))
plt.pcolormesh(x0[:-1], y0, np.log10(h).T, cmap='nipy_spectral')
plt.xlabel('p-order')
plt.ylabel('link length [min]')
plt.xticks(np.arange(3.5, 19.5, 2))
ax = plt.gca()
ax.set_xticklabels(np.arange(3, 19, 2))
plt.colorbar(label='log-number')
#plt.clim([0, int(0.75 * np.nanmax(h))])
#plt.savefig(saveroot + 'po_length_hist2d.png', dpi=200)
#plt.close(fig)