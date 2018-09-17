#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:22:26 2018

@author: Sebastijan Mrak <smrak@bu.edu>
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import nexrad_quickplot as nq

import cartomap as cm
from glob import glob
from dateutil import parser
import h5py
import yaml
from numpy import array, where, ma, isnan, arange, mean, isfinite, mgrid, sort
from numpy import fromfile, float32, linspace, floor, ceil, add, multiply
from numpy import meshgrid, rot90, flip
from datetime import datetime
from scipy import ndimage
from pyGnss import gnssUtils as gu

import concurrent.futures

def _toLuma(x):
    """
    RBG -> Luma conversion
    After https://en.wikipedia.org/wiki/Luma_(video)
    """
    rr = multiply(x[:,:,0], 0.2126)
    gg = multiply(x[:,:,1], 0.7152)
    bb = multiply(x[:,:,2], 0.0722)
    yy = add(rr,gg,bb)
    
    return yy

def returndTEC(fn,dtype='single',darg=1,time='dt'):
    """
    Return a single slice with coordinates from the HDF image collection. Multi
    type query:
        dtype = single:
            darg = i-th element of the array. Must be an integer
            darg = timestamp. It will find the closes time stamp in the collection
            and return the slice with coordinates. Input either datetime.datetime
            or strng which is parsed via parser.parse()
        time = return time format. If dt = posix, else datetime.datetime
    Return:
        time[dt,posix], xgrid, ygrid, image
    """
    def _getIndex(t,t0):
        i = abs(t-t0).argmin()
        return i
    f = h5py.File(fn, 'r')
    xgrid = f['data/xgrid'].value
    ygrid = f['data/ygrid'].value
    t0 = f['data/time'].value
    t = array([datetime.utcfromtimestamp(t) for t in t0])
    if dtype == 'single':
        i = darg
    if dtype == 't':
        if isinstance(darg,str):
            darg = parser.parse(darg)
        elif isinstance(darg, datetime):
            pass
#        elif isinstance(darg,str):
#            try:
#                t0 = parser.parse(darg)
#                i = _getIndex(t,t0)
#            except Exception as e:
#                print(e)
        else:
            raise("'darg' must be datetime or stging type")
    i = _getIndex(t,darg)
    im = f['data/im'][i]
    if time == 'posix':
        t = t0
    return t, xgrid, ygrid, im

def returnNEXRAD(folder, downsample=1, dtype='single',darg='',im_mask=220,RGB=0):
    if dtype == 'single':
        nqr = nq.load(folder + darg, downsample=downsample)
    nqr_lon = nqr.lon
    nqr_lat = nqr.lat
    nqr_im = nqr.values
    if not RGB:
        nqr_im= _toLuma(nqr_im)
        Z = flip(rot90(ma.masked_where((nqr_im>=im_mask),nqr_im),2),1)
    else:
        Z = ma.masked_where((nqr_im>=im_mask),nqr_im)
    X,Y = meshgrid(nqr_lon,nqr_lat)
    
    
    return X,Y,Z

def getNeighbours(image,i,j,N=3):
    """
    Return an array of <=9 neighbour pixel of an image with a center at (i,j)
    """
    nbg = []
    m = int(floor(N/2))
    M = int(ceil(N/2))
    for k in arange(i-m, i+M):
        for l in arange(j-m, j+M):
            try:
                nbg.append(image[k,l])
            except Exception as e:
                pass
    return array(nbg)

def fillPixels(im, N=1):
    """
    Fill in the dead pixels. If a dead pixel has a least 4 finite neighbour
    pixel, than replace the center pixel with a mean valuse of the neighbours
    """
    
    for n in range(N):
        skip = int(floor((3+n)/2))
        for i in arange(0,im.shape[0],skip):
            for j in arange(0,im.shape[1],skip):
                # Check if th epixel is dead, i.e. empty
                if isnan(im[i,j]):
                    # Get its neighbours as a np array
                    nbg = getNeighbours(im,i,j,N=(3+n))
                    # If there are at leas 4 neighbours, replace the value with a mean
                    if sum(isfinite(nbg)) >= 4:
                        ix = where(isfinite(nbg))[0]
                        avg = mean(nbg[ix])
                        im[i,j] = avg
    return im

def getEUVMaskCoordinates(latlim=[-89.5,89.5],lonlim=[-180,180],nlat=180,nlon=360):
    xgrid, ygrid = mgrid[lonlim[0]:lonlim[1]:nlon*1j, latlim[0]:latlim[1]:nlat*1j]
    return xgrid,ygrid

def getEUVMask(time,nlat=180,nlon=360,
               EUVDIR = '/home/smrak/Documents/eclipse/MapsSDOdisk300/'):
    """
    I: time in posix
    """
    xgrid, ygrid = getEUVMaskCoordinates(nlat=nlat, nlon=nlon)
    npts = nlat*nlon
    #Import EUV mask files
    flist = sort(glob(EUVDIR+'*.bin'))
    if isinstance(time, float) or isinstance(time, int):
        Tframe_full = datetime.utcfromtimestamp(time)
    else: 
        Tframe_full = time
    if int(Tframe_full.strftime('%H')) >= 16 and int(Tframe_full.strftime('%H')) < 22:
        # find right filename extension
        TframeHM = Tframe_full.strftime('%H%M')
        flist = sort(glob(EUVDIR+'*'+TframeHM+'.bin'))
        # Get Mask
        data = fromfile(flist[0],count=npts, dtype=float32).reshape((nlat,nlon))
        return xgrid, ygrid, data
    else:
        return 0, 0, 0


#def plotCartoMap(latlim=[0,75],lonlim=[-40,40],parallels=[],meridians=[],
#                 figsize=(12,8),projection='stereo',title='',resolution='110m',
#                 states=True,grid_linewidth=0.5,grid_color='black',terrain=False,
#                 grid_linestyle='--', background_color=None,border_color='k'):
#
#    STATES = cfeature.NaturalEarthFeature(
#            category='cultural',
#            name='admin_1_states_provinces_lines',
#            scale='50m',
#            facecolor='none')
#    if figsize is None:
#        figsize = (12,8)
#    plt.figure(figsize=figsize)
#    if projection == 'stereo':
#        ax = plt.axes(projection=ccrs.Stereographic(central_longitude=(sum(lonlim)/2)))
#    if projection == 'merc':
#        ax = plt.axes(projection=ccrs.Mercator())
#    if projection == 'plate':
#        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=(sum(lonlim)/2)))
#    if background_color is not None:
#        ax.background_patch.set_facecolor(background_color)
#    ax.set_title(title)
#    ax.coastlines(color=border_color,resolution=resolution) # 110m, 50m or 10m
#    if states:
#        ax.add_feature(STATES, edgecolor=border_color)
#    ax.add_feature(cfeature.BORDERS,edgecolor=border_color)
#    if terrain:
#        ax.stock_img()
#    ax.set_extent([lonlim[0], lonlim[1], latlim[0], latlim[1]])
#    
#    if projection != 'merc':
#        gl = ax.gridlines(crs=ccrs.PlateCarree(),color=grid_color,
#                          linestyle=grid_linestyle,linewidth=grid_linewidth)
#    else:
#        gl = ax.gridlines(crs=ccrs.PlateCarree(),color=grid_color,draw_labels=True,
#                          linestyle=grid_linestyle,linewidth=grid_linewidth)
#        gl.xlabels_top=False
#        gl.ylabels_right=False
#    gl.xlocator = mticker.FixedLocator(meridians)
#    gl.ylocator = mticker.FixedLocator(parallels)
#    gl.xlabels_top = False
#    gl.xformatter = LONGITUDE_FORMATTER
#    gl.yformatter = LATITUDE_FORMATTER
#    
#    return ax

def makeImage(im, pixel_iter):
    im = fillPixels(im, pixel_iter)
    im = fillPixels(im)
    im = ndimage.median_filter(im, 3)
    image = ma.masked_where(isnan(im),im)
    return image

def getTotality():
        totality_path = h5py.File('/home/smrak/Documents/eclipse/totality.h5', 'r')
        lat_n = totality_path['path/north_lat'].value
        lon_n = totality_path['path/north_lon'].value
        lat_s = totality_path['path/south_lat'].value
        lon_s = totality_path['path/south_lon'].value
        
        return lon_s, lat_s, lon_n, lat_n
    
def getTotalityCenter(fn='/home/smrak/Documents/eclipse/totality.h5'):
    totality_path = h5py.File(fn, 'r')
    lat_c = totality_path['path/center_lat'].value
    lon_c = totality_path['path/center_lon'].value
    
    return lon_c, lat_c

# Imageinput

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('year', type=str)
    p.add_argument('day', type=str)
    p.add_argument('t0', type=str, help='Processing start time', default='00:00')
    p.add_argument('t1', type=str, help='Processing start time', default='01:00')
    p.add_argument('cfg', type=str)
    p.add_argument('-m', '--cfgmap', type=str, help='Yaml configuration file with the map settings',
                   default='map/example_map.yaml')
    
    P = p.parse_args()

    year = P.year
    day = P.day
    t0 = P.t0
    t1 = P.t1
    
    timelim = [datetime.strptime('{} {} {} 0'.format(year, day, t0),'%Y %j %H:%M %S'),
               datetime.strptime('{} {} {} 0'.format(year, day, t1),'%Y %j %H:%M %S')]

    stream = yaml.load(open(P.cfg, 'r'))
    gpsfn = stream.get('datafn')
    savepath = stream.get('savedir')
    
    fillpixel_iter = stream.get('fillpixel_iter')
    skip = stream.get('skip')
    projection = stream.get('projection')
    latlim = stream.get('latlim')
    lonlim = stream.get('lonlim')
    clim = stream.get('clim')
    
    # Position X
    position = stream.get('position') # [lon,lat]

    # Map limits
    if stream.get('maplatlim') is not None:
        maplonlim = stream.get('maplonlim')
        maplatlim = stream.get('maplatlim')
    else:
        maplonlim = [-135, -65]
        maplatlim = [20, 55]
        
    #Map params
    if stream.get('meridians') is not None:
        parallels = stream.get('parallels')
        meridians = stream.get('meridians')
    else:
        parallels = [10,20,30,40,50,60]
        meridians = [-140,-120,-100,-80,-60]
    
    # Map settings
    mapcfg = P.cfgmap
    streammap = yaml.load(open(mapcfg, 'r'))
    figure_size = streammap.get('figure_size')
    background_color = streammap.get('background_color')
    border_color = streammap.get('border_color')
    grid_color = streammap.get('grid_color')
    grid_linestyle = streammap.get('grid_linestyle')
    grid_linewidth = streammap.get('grid_linewidth')
    terrain = streammap.get('terrain')
    
    # Image params
    image_type = streammap.get('image_type')
    cmap = streammap.get('cmap')
    image_nlevels = streammap.get('image_nlevels')
    
    # Overlays @ eclipse
    totality = streammap.get('totality')
    penumbra = streammap.get('penumbra')
    laplacian = streammap.get('laplacian')
    
    laplacian_levels = streammap.get('laplacian_levels')
    penumbra_levels = streammap.get('penumbra_levels')
    # Marker
    marker = streammap.get('marker')
    marker_color = streammap.get('marker_color')
    marker_size = streammap.get('marker_size')
    marker_width = streammap.get('marker_width')
    
    # GPS Images
    gpsdata = h5py.File(gpsfn, 'r')
    time = gpsdata['data/time'].value
    xgrid = gpsdata['data/xgrid'].value
    ygrid = gpsdata['data/ygrid'].value
    im = gpsdata['data/im'][:][:][:]
    datetimetime = array([datetime.utcfromtimestamp(t) for t in time])
    
    idt = where( (datetimetime >= timelim[0]) & ((datetimetime <= timelim[1])))[0]
    dt = datetimetime[idt]
    iterate1 = arange(idt[0],idt[-1]+1,skip)
    iterate2 = arange(0,len(dt),skip)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        im = [ex.submit(makeImage, im[i], fillpixel_iter) for i in iterate1]
    
    j = 0
    for i in iterate2:
        print ('Plotting figure {}/{}'.format(j+1,iterate2.shape[0]))
        title = dt[i]
        # Get a map
        ax = cm.geogmap(figsize=figure_size, projection=projection,title=title,
                          terrain=terrain, lonlim=lonlim,latlim=latlim,
                          meridians=meridians, parallels=parallels,
                          grid_linewidth=grid_linewidth,grid_color=grid_color,
                          background_color=background_color, border_color=border_color)
        image = im[j].result()
        j+=1
        # Plot image
        try:
            if image_type == 'contourf':
                levels = linspace(clim[0],clim[1], 40)
                image[image<=clim[0]] = levels[0]
                image[image>=clim[1]] = levels[-1]
                cs = plt.contourf(xgrid,ygrid,image.T, levels=levels,cmap=cmap, transform=ccrs.PlateCarree())
                cs.cmap.set_under('b')
                cs.cmap.set_over('r')
            else:
                plt.pcolormesh(xgrid,ygrid,image.T,cmap=cmap, transform=ccrs.PlateCarree())
                
            plt.clim(clim)
            cbar = plt.colorbar(ticks=[clim[0], clim[0]/2, 0, clim[1]/2, clim[1]])
            cbar.set_label('$\Delta$TEC [TECu]')
    
            if totality:
                lon_c, lat_c = getTotalityCenter()
                plt.plot(lon_c, lat_c-1, 'k', lw=1, transform=ccrs.PlateCarree())
            if penumbra:
                cmap1 = colors.LinearSegmentedColormap.from_list("", ['white', 'magenta'])
                try:
                    xgm, ygm, data = getEUVMask(dt[i])
                    if laplacian:
                        data = abs(ndimage.filters.laplace(data))
                        if laplacian_levels is None:
                            laplacian_levels = [0.005,0.035,10]
                        levels = linspace(laplacian_levels[0],laplacian_levels[1],laplacian_levels[2])
                        plt.contour(xgm,ygm,data.T, levels, cmap=cmap1,transform=ccrs.PlateCarree())#, alpha=0.9, norm=colors.PowerNorm(gamma=0.7), 
                        
                    else:
                        if penumbra_levels is not None:
                            penumbra_levels = [0.2,1,40]
                        levels = linspace(penumbra_levels[0],penumbra_levels[1],penumbra_levels[2])
                        lw = 0.5
                        plt.contour(xgm,ygm,data.T, levels, colors='w', linewidths=lw, transform=ccrs.PlateCarree())
                except:
                    pass
            # Marker
            if position is not None:
                try:
                    plt.plot(position[0],position[1], marker, c=marker_color, ms=marker_size, mew=marker_width, transform=ccrs.PlateCarree())
                except:
                    print ('Couldnt plot the marker')
            ax.set_extent([maplonlim[0], maplonlim[1],
                           maplatlim[0], maplatlim[1]],crs=ccrs.PlateCarree())
            ax.set_aspect('auto')
        except Exception as e:
            print (e)
        tit = int(gu.datetime2posix([dt[i]])[0])
        plt.savefig(savepath+str(tit)+'.png', dpi=200)
        plt.close()

