#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:06:10 2018

@author: smrak
"""

import cartopy.crs as ccrs
import os
import cartomap.geogmap as cm
import h5py
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import array, multiply, add, ma, rot90, flip, linspace, meshgrid
import gps2d

# CFG MAP
wdir = os.getcwd()
import platform
if platform.system() == 'Linux':
    # Imgage filename path
    folder = '/media/smrak/Eclipse2017/Eclipse/hdf/eclipse/'
    # NEXRAD image 
    nexradfolder = '/media/smrak/Eclipse2/nexrad/233'
    # Totality data
    totalityfn = '/home/smrak/Documents/eclipse/totality.h5'
elif platform.system() == 'Windows':
    # Imgage filename path
    folder = 'E:\\'
    # NEXRAD image 
    nexradfolder = 'E:\\nexrad\\233\\'
    # Totality data
    totalityfn = 'E:\\totality.h5'
# What to do?
DTEC = 1
NEXRAD = 1
TOTALITY = 1
dtype = 'single'
save = 'E:\\grlfigs\\'
save = ''

nqimage = '/nexrad2017-08-20T13-00-00.png'
nqimage = '/nexrad2017-08-21T18-00-00.png'
    
# Map settings
mapcfg = wdir + '/map/example_map.yaml'
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
# --------------------------------------------------------------------------- #
# Make a map: ALWAYS
# --------------------------------------------------------------------------- #
fig = cm.plotCartoMap(projection=projection,
                     latlim=latlim,lonlim=lonlim,parallels=parallels,
                     meridians=meridians,figsize=figure_size,
                     background_color=background_color,border_color=border_color,
                     grid_color=grid_color,grid_linestyle=grid_linestyle,
                     grid_linewidth=grid_linewidth, terrain=terrain)
# --------------------------------------------------------------------------- #
# Plot the TEC image
# --------------------------------------------------------------------------- #
if DTEC:
    # Get data
    datafn = 'single233_02_130_60.h5'
#    datafn = 'single232_02_130_60.h5'
    #datafn = 'single233_02_cut.h5'
    fn = folder + datafn
    #i = 2273 # 18:57
    i = 2243 # 18:42
    #i = 2233 # 18:37:00
    #i = 2213 # 18:27
#    i = 1679 # 14:00
#    i = 1559 # 13:00
#    i = 1619 # 13:30
    t, xgrid, ygrid, tec = gps2d.returndTEC(fn, dtype=dtype, darg=i)
    if fillpixeriter > 0:
        tec = gps2d.makeImage(tec,pixel_iter=fillpixeriter)
    # Plot data
    plt.title(t[i])
    if image_type == 'contourf':
        tec[tec<=clim[0]] = levels[0]
        tec[tec>=clim[1]] = levels[-1]
        plt.contourf(xgrid,ygrid,tec.T,levels,cmap=cmap,transform=ccrs.PlateCarree())
    else:
        plt.pcolormesh(xgrid,ygrid,tec.T,cmap=cmap,transform=ccrs.PlateCarree())
    plt.clim(clim)
    # Fig utils
    cbar = plt.colorbar(ticks=[clim[0], clim[0]/2, 0, clim[1]/2, clim[1]])
    cbar.set_label('$\Delta$TEC [TECu]')
if NEXRAD:
    X, Y, Z = gps2d.returnNEXRAD(nexradfolder, downsample=16, darg=nqimage)
    plt.contourf(X,Y,Z,10,cmap='Greys_r',
                transform=ccrs.PlateCarree())
if TOTALITY:
    lon_t, lat_t = gps2d.getTotalityCenter(totalityfn)
    plt.plot(lon_t, lat_t-1, '--k', lw=2, transform=ccrs.PlateCarree())
    
if save != '':
    try:
        filename = datetime.strftime(t[i], '%Y%d%m-%H%M%S')
        if NEXRAD:
            filename+='_nexrad'
        figname = save + filename + '.png'
        plt.tight_layout()
        plt.savefig(figname,dpi=300)
        plt.close(fig)
    except Exception as e:
        print(e)
        
