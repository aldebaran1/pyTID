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

fn = '/media/smrak/gnss/hdf/2013_0215T0000-0216T0000_conus046.yaml_20el_60s.h5'

D = h5py.File(fn, 'r')
po = D['po'][:]
po_length = D['po_length'][:]

po_length[po_length >= 8*60] = np.nan

fig = plt.figure(figsize=[8,5])
plt.title('Pool size: {}'.format(po.size))
plt.hist(list(po), range=(4,20), bins=16, density=True, color='b', align='left', rwidth=0.9)
plt.xlabel('Polynomial order')
plt.ylabel('Probability')
#plt.xlim([3,polynom_list[-1]])
#plt.savefig(saveroot + 'po_hist.png', dpi=100)
#plt.close(fig)

fig = plt.figure(figsize=[8,5])
plt.title('Pool size: {}'.format(po.size))
plt.hist(list(po_length), bins=50, density=True, color='b', align='left', rwidth=0.9)
plt.xlabel('LOS length [min]')
plt.ylabel('Probability')
#plt.savefig(saveroot + 'polength_hist.png', dpi=100)
#plt.close(fig)

h, x0, y0 = np.histogram2d(po, po_length, bins=[16,50], range=[[4,20], [0,500]])
fig = plt.figure(figsize=[8,5])
plt.pcolormesh(x0[:-1], y0, np.log10(h).T, cmap='jet')
#h, x0, y0, im = plt.hist2d(po, po_length, bins=[16,50], range=[[4,20], [0,500]], norm=colors.LogNorm(vmin=0.01))
plt.xlabel('Polynomial order')
plt.ylabel('LOS length [min]')
plt.colorbar()
#plt.clim([0, int(0.75 * np.nanmax(h))])
#plt.savefig(saveroot + 'po_length_hist2d.png', dpi=200)
#plt.close(fig)