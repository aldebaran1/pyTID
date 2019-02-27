#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:06:10 2018

@author: smrak
"""

import cartopy.crs as ccrs
import os
import cartomap.geogmap as cm
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import linspace, meshgrid
import gps2d


if 'title' in vars():
    del title

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
RGB = 0
DTEC = 1
NEXRAD = 0
TOTALITY = 1
MASK = 1
T = '2017-8-22T04:30'
T = '2017-8-21T17:30'
T1 = '2017-8-21T16:00'
T2 = '2017-8-21T20:00'
#T = '2017-8-20T14:00'
#T = 
dtype = 'single'
#save = 'E:\\surav\\'
save = 'E:\\grlfigs\\cover\\'
save = ''

#nqimage = '/nexrad2017-08-20T13-00-00.png'
nqimage = '/nexrad2017-08-21T18-00-00.png'
#nqimage = '/nexrad2017-08-21T14-00-00.png'
    
# Map settings
cfgdir = 'E:\\eclipse\\euv\\'
mapcfg = cfgdir + 'map.yaml'
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
# --------------------------------------------------------------------------- #
# Make a map: ALWAYS
# --------------------------------------------------------------------------- #
fig = cm.plotCartoMap(projection=projection,
                     latlim=latlim,lonlim=lonlim,parallels=parallels,
                     meridians=meridians,figsize=figure_size,
                     background_color=background_color,border_color=border_color,
                     grid_color=grid_color,grid_linestyle=grid_linestyle,
                     grid_linewidth=grid_linewidth, terrain=terrain,figure=True)
# --------------------------------------------------------------------------- #
# Plot the TEC image
# --------------------------------------------------------------------------- #
if DTEC:
    # Get data
    datafn = 'single233_02_130_60.h5'
#    datafn = 'single232_02_130_60.h5'
#    datafn = 'single233_02_cut.h5'
#    datafn = 'single234_02.h5'
    fn = folder + datafn
    #i = 2273 # 18:57
#    i = 2243 # 18:42
    i = 2235 # 18:38
#    i = 2213 # 18:27
#    i = 1679 # 14:00
#    i = 1559 # 13:00
    i = 1619 # 13:30
    i = 1799
    t, xgrid, ygrid, tec = gps2d.returndTEC(fn, dtype='treq', darg = [T1,T2])

    if fillpixeriter > 0:
        tec = gps2d.makeImage(tec,pixel_iter=3)
    # Plot data
    if T == '':
        title = 'TEC: ' + str(t[i])
    else:
        title = 'TEC: ' + T
    if image_type == 'contourf':
        tec[tec<=clim[0]] = levels[0]
        tec[tec>=clim[1]] = levels[-1]
        tidim = plt.contourf(xgrid,ygrid,tec.T,levels,cmap=cmap,transform=ccrs.PlateCarree())
    elif image_type == 'scatter':
        X = meshgrid(xgrid, ygrid)
        tidim = plt.scatter(X[0], X[1], c=tec.T,
                            s=8, cmap=cmap, 
                            transform=ccrs.PlateCarree())
    else:
        tidim = plt.pcolormesh(xgrid,ygrid,tec.T,cmap=cmap,transform=ccrs.PlateCarree())
    plt.clim(clim)
    # Fig utils
    cbar = plt.colorbar(mappable=tidim, #cax=cax,
                        ticks=[clim[0], clim[0]/2, 0, clim[1]/2, clim[1]])
    cbar.set_label('$\Delta$TEC [TECu]')

if MASK:
    from dateutil import parser
    from pyGnss import gnssUtils
    dt = parser.parse(T)
    tposix = gnssUtils.datetime2posix([dt])
    mask_x, mask_y, mask = gps2d.getEUVMask(tposix[0],
            EUVDIR='C:\\Users\\smrak\\Google Drive\BU\\Projects\\Eclipse2017\\data\\Drob\\HiResFull300\\')
    penumbra_levels = [0.2,1,40]
    levels = linspace(penumbra_levels[0],penumbra_levels[1],penumbra_levels[2])
    lw = 1
    plt.contour(mask_x, mask_y, mask.T, levels, colors='w', #cmap='Greys_r',#colors='w', 
                linewidths=lw, transform=ccrs.PlateCarree())

if NEXRAD:
    if 'title' in vars():
        title += '\n NEXRAD: ' + nqimage[7:-4]
    else:
        title = 'NEXRAD: ' + nqimage[7:-4]
    X, Y, Z = gps2d.returnNEXRAD(nexradfolder, downsample=16, darg=nqimage, RGB=RGB)
    if not RGB:
        plt.contourf(X,Y,Z,15,cmap='Greys_r',transform=ccrs.PlateCarree())
    else:
        if not projection == 'stereo':
            plt.imshow(Z,origin='upper',
                         extent=[X[0][0], X[0][-1], Y[0][0], Y[-1][0]],
                         transform=ccrs.PlateCarree())
        else:
            print (projection + ' projection doest work for RGB NEXRAD images')
        
if TOTALITY:
    lon_t, lat_t = gps2d.getTotalityCenter(totalityfn)
    plt.plot(lon_t, lat_t-1, '--k', lw=2, transform=ccrs.PlateCarree())
    
    

if marker_locations is not None:
    if len(marker_locations) > 0 :
        for loc in marker_locations:
            plt.scatter(loc[0],loc[1],
                        marker=marker,s=marker_size,
                        c=marker_color,lw=marker_width,
                        transform=ccrs.PlateCarree())

# Make title:
plt.title(title)
if save != '':
    try:
        filename = datetime.strftime(t[i], '%Y%d%m-%H%M%S')
        if NEXRAD:
            filename+='_nexrad'
        if marker_locations is not None:
            filename += '_markers'
        if MASK:
            filename += '_mask'
        figname = save + filename + '.png'
        plt.savefig(figname, dpi=300)
        plt.close(fig)
        print ('Figure saved.')
    except Exception as e:
        print(e)
else:
    plt.show()
        
