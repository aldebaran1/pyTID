#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:22:26 2018

@author: Sebastijan Mrak <smrak@bu.edu>
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from cartomap import geogmap as gm
from glob import glob
from dateutil import parser
import h5py, os
import yaml
from numpy import array, where, ma, isnan, arange, mean, isfinite, mgrid, sort, ones
from numpy import fromfile, float32, linspace, floor, ceil, add, multiply, copy
from numpy import meshgrid, rot90, flip, ndarray
from datetime import datetime
from scipy import ndimage

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
    im = f['data/im']
    
    if dtype == 'single':
        i = darg
        im = f['data/im'][i]
    elif dtype == 't':
        if isinstance(darg,str):
            darg = parser.parse(darg)
        elif isinstance(darg, datetime):
            pass
        else:
            raise("'darg' must be datetime or stging type")
        i = _getIndex(t, darg)
        im = f['data/im'][i]
    elif dtype == 'treq':
        if isinstance(darg, (list, ndarray)):
            darg = [parser.parse(d) for d in darg]
        elif isinstance(darg[0], datetime):
            pass
        else:
            raise("'darg' must be datetime or stging type")
        i1 = _getIndex(t, darg[0])
        i2 = _getIndex(t, darg[1])
        im = f['data/im'][i1:i2]
        t = t[i1:i2]
    if time == 'posix':
        t = t0
    return t, xgrid, ygrid, im

def returnNEXRAD(folder, downsample=1, dtype='single',darg='',im_mask=220, RGB=0):
    import nexrad_quickplot as nq
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
    X = im.shape[0]-1
    Y = im.shape[1]-1
    imcopy = copy(im)
    for n in range(N):
        skip = int(floor((3+n)/2))
        starti = 0
        startj = 0
        forwardi = int(floor(0.6*X))
        backwardi = int(floor(0.4*X))
        if n%2 == 0:
            for i in arange(starti, forwardi, skip):
                for j in arange(startj, Y, skip):
                    # Check if th epixel is dead, i.e. empty
                    if isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(isfinite(nbg)) >= 4:
                            ix = where(isfinite(nbg))[0]
                            avg = mean(nbg[ix])
                            im[i,j] = avg
            for i in arange(X, backwardi, -skip):
                for j in arange(Y, 0, -skip):
                    # Check if th epixel is dead, i.e. empty
                    if isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(isfinite(nbg)) >= 4:
                            ix = where(isfinite(nbg))[0]
                            avg = mean(nbg[ix])
                            im[i,j] = avg
        else:
            for j in arange(startj, Y, skip):
                for i in arange(starti, forwardi, skip):
                    # Check if th epixel is dead, i.e. empty
                    if isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(isfinite(nbg)) >= 4:
                            ix = where(isfinite(nbg))[0]
                            avg = mean(nbg[ix])
                            im[i,j] = avg
            
            for j in arange(Y, 0, -skip):
                for i in arange(X, backwardi, -skip):
                    # Check if th epixel is dead, i.e. empty
                    if isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
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

def makeImage(im, pixel_iter):
    im = fillPixels(im, pixel_iter)
    im = fillPixels(im)
    im = ndimage.median_filter(im, 3)
    #im = ma.masked_where(isnan(im),im)
    return im

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
    p.add_argument('file', type=str, help='Input HDF5 file')
    p.add_argument('--t0', type=str, help='Processing start time yyyy-mm-dd', default=None)
    p.add_argument('--t1', type=str, help='Processing start time yyyy-mm-dd', default=None)
    p.add_argument('--cfg', type=str)
    p.add_argument('--odir', type=str, help='Output directory', default=None)
    p.add_argument('-m', '--cfgmap', type=str, help='Yaml configuration file with the map settings',
                   default='map/example_map.yaml')
    
    P = p.parse_args()
    
    assert P.file.endswith('.h5')
    gpsfn = P.file
    
    t0 = parser.parse(P.t0)
    t1 = parser.parse(P.t1)
    
    try:
        stream = yaml.load(open(P.cfg, 'r'))
    except:
        stream = yaml.load(open(os.path.join(os.getcwd(), P.cfg), 'r'))
    
    fillpixel_iter = stream.get('fillpixel_iter')
    skip = stream.get('skip')
    projection = stream.get('projection')
    latlim = stream.get('latlim')
    lonlim = stream.get('lonlim')
    clim = stream.get('clim')
    cmap = stream.get('cmap')
    # Coordinates' lines
    parallels = stream.get('parallels')
    meridians = stream.get('meridians')
    mag_parallels = stream.get('mag_parallels')
    mag_meridians = stream.get('mag_meridians')
    mlon_cs = stream.get('mlon_cs')
    if (mag_parallels is not None) or (mag_meridians is not None):
        apex = True
    else:
        apex = False
    # Map settings
    mapcfg = P.cfgmap
    try:
        streammap = yaml.load(open(mapcfg, 'r'))
    except:
        streammap = yaml.load(open(os.path.join(os.getcwd(), mapcfg), 'r'))
    figure_size = streammap.get('figure_size')
    background_color = streammap.get('background_color')
    border_color = streammap.get('border_color')
    grid_color = streammap.get('grid_color')
    grid_linestyle = streammap.get('grid_linestyle')
    grid_linewidth = streammap.get('grid_linewidth')
    terrain = streammap.get('terrain')
    states = streammap.get('states')
    # Image params
    image_type = streammap.get('image_type')
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
    try:
        altkm = gpsdata.attrs['altkm']
    except:
        altkm = 0
    
    
    datetimetime = array([datetime.utcfromtimestamp(t) for t in time])
    dirnametime = [0].strftime('%y%m%d')
    if t0 is not None and t1 is not None:
        timelim = [t0, t1]
        idt = ones(datetimetime.size, dtype=bool)
    else:
        idt = where( (datetimetime >= timelim[0]) & ((datetimetime <= timelim[1])))[0]
    
    dt = datetimetime[idt]
    iterate1 = arange(idt[0], idt[-1]+1, skip)
    iterate2 = arange(0, dt.size, skip)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        im = [ex.submit(makeImage, im[i], fillpixel_iter) for i in iterate1]
    
    j = 0
    for i in iterate2:
        print ('Plotting figure {}/{}'.format(j+1,iterate2.shape[0]))
        # Get a map
        fig, ax = gm.plotCartoMap(figsize=figure_size, projection=projection, #title=dt[i],
                          terrain=terrain, states=states, border_color=border_color,
                          background_color=background_color, 
                          lonlim=lonlim,latlim=latlim,
                          title="{}, alt = {} km".format(dt[i], altkm),
                          meridians=meridians, parallels=parallels,
                          grid_linewidth=grid_linewidth,grid_color=grid_color,
                          apex=apex, mlon_cs=mlon_cs, date=dt[i],
                          mlon_levels=mag_meridians, mlat_levels=mag_parallels
                          )
        image = im[j].result()
        j+=1
        # Plot image
        try:
            if image_type == 'contourf':
                levels = linspace(clim[0],clim[1], 40)
                image[image<=clim[0]] = levels[0]
                image[image>=clim[1]] = levels[-1]
                imax = plt.contourf(xgrid,ygrid,image.T, levels=levels,cmap=cmap, transform=ccrs.PlateCarree())
                imax.cmap.set_under('b')
                imax.cmap.set_over('r')
            else:
                imax = plt.pcolormesh(xgrid,ygrid,image.T,cmap=cmap, transform=ccrs.PlateCarree())
                
            plt.clim(clim)
#            cbar = plt.colorbar()
#            cbar.set_label('$\Delta$TEC [TECu]')
            posn = ax.get_position()
            cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
            fig.colorbar(imax, cax=cax, label='TEC [TECu]',ticks=[clim[0], clim[0]/2, 0, clim[1]/2, clim[1]])
    
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
#            if position is not None:
#                try:
#                    plt.plot(position[0],position[1], marker, c=marker_color, ms=marker_size, mew=marker_width, transform=ccrs.PlateCarree())
#                except:
#                    print ('Couldnt plot the marker')
#            ax.set_extent([maplonlim[0], maplonlim[1],
#                           maplatlim[0], maplatlim[1]],crs=ccrs.PlateCarree())
#            ax.set_aspect('auto')
        except Exception as e:
            print (e)
            
        # Save
        
        odir = P.odir if P.odir is not None else '/media/smrak/gnss/images/{}/'.format(dirnametime+'_'+str(int(altkm)))
        if not os.path.exists(odir):
            import subprocess
            subprocess.call('mkdir -p {}'.format(odir), shell=True, timeout=2)
        tit = dt[i].strftime('%m%d_%H%M')
        plt.savefig(odir+str(tit)+'.png', dpi=150)
        plt.close()

