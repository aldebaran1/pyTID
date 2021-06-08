# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:24:08 2020

@author: smrak@bu.edu
"""

from gpstec import gpstec
from datetime import datetime
from cartomap import geogmap as gm
import numpy as np
import cartopy.crs as ccrs
from scipy.ndimage import laplace

latlim = [-10, 70]
lonlim = [-150, -30]
projection = 'stereo'
cmap = 'gray'
clim = [0, 30]



dr = 'G:\\My Drive\\scintillation_data\\tid\\20170528\\'

TECFN = [dr + 'conv_20170527T0000-20170528T0000.h5',
         dr + 'conv_20170528T0000-20170529T0000.h5']

skip = 3
average = 5

it = datetime(2017, 5, 28, 3, 0)
D = gpstec.merge_time(TECFN)
tectime = D['time']

xgrid = D['xgrid']
ygrid = D['ygrid']
idx = (xgrid >= -125) & (xgrid <= -58)
idy = (ygrid >= 25) & (ygrid <= 50)
idt_tec = abs(tectime - it).argmin()
tecim = np.nanmean(D['tecim'][idt_tec-average : idt_tec+1], axis=0)
tecim[~idx, :] = np.nan
tecim[:, ~idy] = np.nan

fig, ax = gm.plotCartoMap(latlim=latlim, lonlim=lonlim, projection='stereo',
                          meridians=None, parallels=None,
                          grid_linewidth=1, states = False,
                          title=it, background_color='grey',
                          apex=True, mlat_levels=[-20,0,20,40,60,80,90],
                          mlat_colors='w', mgrid_width=1, mgrid_style='--',
                          mlon_levels=np.arange(0,361,40), mlat_labels=False,
                          mlon_colors='w', mlon_labels=False)

im = ax.pcolormesh(xgrid, ygrid, tecim.T, cmap=cmap, vmin=clim[0], vmax=clim[1],
                      transform=ccrs.PlateCarree())

lap = laplace(tecim)
lap[lap<0] = np.nan
ax.contour(xgrid, ygrid, lap.T, cmap='jet', transform=ccrs.PlateCarree())