#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:22:26 2018

@author: Sebastijan Mrak <smrak@bu.edu>
"""

import cartopy.crs as ccrs
import os
import cartomap.geogmap as cm
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import linspace, arange, ndarray
import gps2d

# CFG MAP
wdir = os.getcwd()
import platform
if platform.system() == 'Linux':
    # Imgage filename path
    folder = '/media/smrak/Eclipse2017/Eclipse/hdf/eclipse/'
elif platform.system() == 'Windows':
    # Imgage filename path
    folder = 'E:\\'
# What to do?
#dtype = 'all'
dtype = 'treq'
save = 'E:\\eclipse\\euv\\'
fmcfg = 'E:\\eclipse\\euv\\'
# Map settings
mapcfg = fmcfg + 'map.yaml'
streammap = yaml.load(open(mapcfg, 'r'))
projection = streammap.get('projection')
figure_size = streammap.get('figure_size')
background_color = streammap.get('background_color')
border_color = streammap.get('border_color')
grid_color = streammap.get('grid_color')
grid_linestyle = streammap.get('grid_linestyle')
grid_linewidth = streammap.get('grid_linewidth')
terrain = streammap.get('terrain')
# Map limits
lonlim = streammap.get('lonlim')
latlim = streammap.get('latlim')
#Map params
parallels = streammap.get('parallels')
meridians = streammap.get('meridians')
# Image settings
image_type = streammap.get('image_type')
cmap = streammap.get('cmap')
clim = streammap.get('clim')
im_levels = streammap.get('image_nlevels')
levels = linspace(clim[0],clim[1], im_levels)
# Image processing
fillpixeriter = 3
# Image overlays: Scatter locations
marker_locations = streammap.get('marker_location')
marker = streammap.get('marker')
marker_color = streammap.get('marker_color')
marker_size = streammap.get('marker_size')
marker_width = streammap.get('marker_width')

#datafn = 'single234_02.h5'
skip = 3
datafn = 'single233_02_130_60.h5'
totalityfn = 'E:\\totality.h5'
T1 = '2017-8-21T15:30'
T2 = '2017-8-21T20:30'

fn = folder + datafn
t, xgrid, ygrid, tec = gps2d.returndTEC(fn, darg=[T1, T2], dtype=dtype)
MASK = True
TOTALITY = True

iterate = arange(0, t.shape[0], skip)
for i in iterate:
    # New figure
    fig = cm.plotCartoMap(projection=projection,
                     latlim=latlim,lonlim=lonlim,parallels=parallels,
                     meridians=meridians,figsize=figure_size,
                     background_color=background_color,border_color=border_color,
                     grid_color=grid_color,grid_linestyle=grid_linestyle,
                     grid_linewidth=grid_linewidth, terrain=terrain,figure=True)
    if fillpixeriter > 0:
        IM = gps2d.makeImage(tec[i],pixel_iter=fillpixeriter)
    if image_type == 'contourf':
        IM[IM <= clim[0]] = levels[0]
        IM[IM >= clim[1]] = levels[-1]
        tidim = plt.contourf(xgrid,ygrid,IM.T,levels,cmap=cmap,transform=ccrs.PlateCarree())
    else:
        tidim = plt.pcolormesh(xgrid,ygrid,IM.T,cmap=cmap,transform=ccrs.PlateCarree())
    plt.clim(clim)
    # Fig utils
    cbar = plt.colorbar(mappable=tidim, #cax=cax,
                        ticks=[clim[0], clim[0]/2, 0, clim[1]/2, clim[1]])
    cbar.set_label('$\Delta$TEC [TECu]')
    
    if MASK:
        from pyGnss import gnssUtils
#        dt = parser.parse(t[i])
        tposix = gnssUtils.datetime2posix([t[i]])
        mask_x, mask_y, mask = gps2d.getEUVMask(tposix[0],
                EUVDIR='C:\\Users\\smrak\\Google Drive\BU\\Projects\\Eclipse2017\\data\\Drob\\HiResFull300\\')
        if isinstance(mask, ndarray):
            penumbra_levels = [0.2,1,50]
            levels1 = linspace(penumbra_levels[0],penumbra_levels[1],penumbra_levels[2])
            lw = 1
            plt.contour(mask_x, mask_y, mask.T, levels1, colors='w', #cmap='Greys_r',#colors='w', 
                        linewidths=lw, transform=ccrs.PlateCarree())
    if TOTALITY:
        lon_t, lat_t = gps2d.getTotalityCenter(totalityfn)
        plt.plot(lon_t, lat_t-1, '--k', lw=2, transform=ccrs.PlateCarree())
    title = 'dTEC: '+ str(t[i])
    # Make title:
    plt.title(title)
    if save != '':
        try:
            filename = datetime.strftime(t[i], '%Y%d%m-%H%M%S')
            if marker_locations is not None:
                filename += '_markers'
            figname = save + filename + '.png'
            plt.tight_layout()
            plt.savefig(figname, dpi=100)
            plt.close(fig)
            print ('Figure saved.')
        except Exception as e:
            print(e)
    else:
        plt.show()
#    break
