#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:06:10 2018

@author: smrak
"""
import nexrad_quickplot as nq
import cartopy.crs as ccrs
import os
import cartomap.geogmap as cm
import h5py
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import array, multiply, add, ma, rot90, flip, linspace, meshgrid
import gps2d

def _toLuma(x):
    rr = multiply(x[:,:,0], 0.2126)
    gg = multiply(x[:,:,1], 0.7152)
    bb = multiply(x[:,:,2], 0.0722)
    yy = add(rr,gg,bb)
    
    return yy

# CFG MAP
wdir = os.getcwd()
# Imgage filename path
folder = '/media/smrak/Eclipse2017/Eclipse/hdf/eclipse/'
# NEXRAD image 
nexradfolder = '/media/smrak/Eclipse2/nexrad/233'

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
# ----------------------------GET IMAGE DATA--------------------------------- #
datafn = 'single233_02_130_60.h5'
#datafn = 'single233_02_cut.h5'
fn = folder + datafn
f = h5py.File(fn, 'r')
xgrid = f['data/xgrid'].value
ygrid = f['data/ygrid'].value
t = f['data/time'].value
i = 2243
#i = 1700
im = f['data/im'][i]
dt = array([datetime.utcfromtimestamp(t) for t in t])
# ---------------------------NEXRAD DATA------------------------------------- #
nqimage = '/nexrad2017-08-21T18-00-00.png'
nqr = nq.load(nexradfolder+nqimage, downsample=16)
nqr_lon = nqr.lon
nqr_lat = nqr.lat
nqr_im = nqr.values
nqr_gs = _toLuma(nqr_im)
X,Y = meshgrid(nqr_lon,nqr_lat)
#z = flip(rot90(ma.masked_where(nqr_im[:,:,0]>=230,nqr_gs),2),1)
z = flip(rot90(ma.masked_where(nqr_gs>=230,nqr_gs),2),1)
# --------------------------------------------------------------------------- #
if fillpixeriter > 0:
    im = gps2d.makeImage(im,pixel_iter=fillpixeriter)
# Make a map
ax = cm.plotCartoMap(projection=projection,title=dt[i],
                     latlim=latlim,lonlim=lonlim,parallels=parallels,
                     meridians=meridians,figsize=figure_size,
                     background_color=background_color,border_color=border_color,
                     grid_color=grid_color,grid_linestyle=grid_linestyle,
                     grid_linewidth=grid_linewidth, terrain=terrain)
# Plot the TEC image

if image_type == 'contourf':
    im[im<=clim[0]] = levels[0]
    im[im>=clim[1]] = levels[-1]
    plt.contourf(xgrid,ygrid,im.T,levels,cmap=cmap,transform=ccrs.PlateCarree())
else:
    plt.pcolormesh(xgrid,ygrid,im.T,cmap=cmap,transform=ccrs.PlateCarree())
plt.clim(clim)
# Fig utils
cbar = plt.colorbar(ticks=[clim[0], clim[0]/2, 0, clim[1]/2, clim[1]])
cbar.set_label('$\Delta$TEC [TECu]')

#plt.contour(X,Y,z,10,cmap='Greys_r',
#           transform=ccrs.PlateCarree())
