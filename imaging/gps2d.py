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
import yaml, platform
from numpy import array, where, ma, isnan, arange, mean, isfinite, mgrid, sort, ones
from numpy import fromfile, float32, linspace, floor, ceil, add, multiply, copy
from numpy import meshgrid, rot90, flip, ndarray, squeeze, nan, divide, isin
from numpy.ma import masked_invalid
from datetime import datetime, timedelta
from scipy import ndimage
from typing import Union
from gpstec import gpstec
from scipy.interpolate import griddata

import concurrent.futures

def interpolateTEC(im: Union[list, ndarray] = None,
                          x0 = None, y0 = None,
                          xgrid = None, ygrid = None,
                          N: int = 512, res=1,
                          method: str = 'linear'):
    assert im is not None, 'Invalid input argument. Has to be a list or np.ndarray with a length of at least 1'
    if x0 is None or y0 is None:
        x0, y0 = meshgrid(arange(im.shape[0]), arange(im.shape[1]))
    x0 = x0.T
    y0 = y0.T
    mask = masked_invalid(im)
    x0 = x0[~mask.mask]
    y0 = y0[~mask.mask]
    X = im[~mask.mask]
    if xgrid is None or ygrid is None:
        xgrid, ygrid = meshgrid(arange(0, im.shape[0], res), 
                                arange(0, im.shape[1], res))
    xgrid = xgrid.T
    ygrid = ygrid.T
    z = griddata((x0,y0), X.ravel(), (xgrid, ygrid), method=method, fill_value=nan)
    return z

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
            except:
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
    if len(im.shape) == 2:
        im = fillPixels(im, pixel_iter)
        im = fillPixels(im)
        im = ndimage.median_filter(im, 3)
    elif len(im.shape) == 3:
        ims = nan * copy(im)
        for i in range(im.shape[0]):
            im0 = fillPixels(im[i], pixel_iter)
            im0 = fillPixels(im0)
            ims[i] = ndimage.median_filter(im0, 3)
        im = mean(ims, axis=0)

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
    p.add_argument('--tlim', type=str, help='Processing time; start,end', default=None, nargs=2)
    p.add_argument('--cfg', type=str)
    p.add_argument('--skip', type=int, default=None)
    p.add_argument('--odir', type=str, help='Output directory', default=None)
    p.add_argument('-m', '--cfgmap', type=str, help='Yaml configuration file with the map settings',
                   default='map/example_map.yaml')
    p.add_argument('--clim', type=float, nargs=2, default=None)
    p.add_argument('--average', type=int, default=1)
    p.add_argument('--projection', type=str, default=None)
    p.add_argument('--cmap', type=str, default=None)
    p.add_argument('--latlim', type=float, nargs=2, default=None)
    p.add_argument('--lonlim', type=float, nargs=2, default=None)
    p.add_argument('--tec', type=str, help='TEC file', default=None)

    P = p.parse_args()

    assert P.file.endswith('.h5')
    gpsfn = P.file

    try:
        stream = yaml.load(open(P.cfg, 'r'), Loader=yaml.SafeLoader)
    except:
        stream = yaml.load(open(os.path.join(os.getcwd(), P.cfg), 'r'), Loader=yaml.SafeLoader)
    
    fntec = P.tec if P.tec is not None else None
    
    fillpixel_iter = stream.get('fillpixel_iter')
    skip = P.skip if (P.skip is not None) else stream.get('skip')
    projection = P.projection if (P.projection is not None) else stream.get('projection')
    latlim = P.latlim if (P.latlim is not None) else stream.get('latlim')
    lonlim = P.lonlim if (P.lonlim is not None) else stream.get('lonlim')
    clim = P.clim if (P.clim is not None) else stream.get('clim')
    cmap = P.cmap if (P.cmap is not None) else stream.get('cmap')
    # Coordinates' lines
    parallels = stream.get('parallels')
    meridians = stream.get('meridians')
    mag_parallels = stream.get('mag_parallels')
    mag_meridians = stream.get('mag_meridians')
    mlon_cs = stream.get('mlon_cs')
    nightshade = stream.get('nightshade')
    if (mag_parallels is not None) or (mag_meridians is not None):
        apex = True
    else:
        apex = False
    # Map settings
    mapcfg = P.cfgmap
    try:
        streammap = yaml.load(open(mapcfg, 'r'), Loader=yaml.SafeLoader)
    except:
        streammap = yaml.load(open(os.path.join(os.getcwd(), mapcfg), 'r'), Loader=yaml.SafeLoader)
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
    
    #Averaging
    average = P.average if (P.average is not None) else 1
    # GPS Images
    gpsdata = h5py.File(gpsfn, 'r')
    time = gpsdata['data/time'][:]
    xgrid = gpsdata['data/xgrid'][:]
    ygrid = gpsdata['data/ygrid'][:]
    im = gpsdata['data/im'][:][:][:]
    gpsdata.close()
    xg, yg = meshgrid(xgrid, ygrid)
    try:
        altkm = gpsdata.attrs['altkm']
    except:
        altkm = int(os.path.split(gpsfn)[1][-13:-10])

    datetimetime = array([datetime.utcfromtimestamp(t) for t in time])
    dirdatetime = datetimetime[0].strftime('%Y%m%d')
    today = datetime.now().strftime('%Y%m%d')
    if P.tlim is not None:
        if today == parser.parse(P.tlim[0]).strftime('%Y%m%d'):
            t0 = parser.parse(dirdatetime + 'T' + P.tlim[0])
        else:
            t0 = parser.parse(P.tlim[0])
        if today == parser.parse(P.tlim[1]).strftime('%Y%m%d'):
            t1 = parser.parse(dirdatetime + 'T' + P.tlim[1])
        else:
            t1 = parser.parse(P.tlim[0])
        timelim = [t0, t1]
        idt = (datetimetime >= timelim[0]) & (datetimetime <= timelim[1])
    else:
        idt = ones(datetimetime.size, dtype=bool)
    dt = datetimetime[idt]
        
    iterate1 = arange(where(idt==1)[0][0], where(idt==1)[0][-1]+1, skip)
    iterate2 = arange(0, dt.size, skip)
    
    if fntec is not None:
        assert os.path.exists(fntec)
        TEC = gpstec.readFromHDF(fntec)
        idttec = (TEC['time'] >= timelim[0]) & (TEC['time'] <= timelim[1])
        idx = (TEC['xgrid'] >= xgrid.min()) & (TEC['xgrid'] <= xgrid.max())
        idy = (TEC['ygrid'] >= ygrid.min()) & (TEC['ygrid'] <= ygrid.max())
        xgtec, ygtec = meshgrid(TEC['xgrid'][idx], TEC['ygrid'][idy])
        idttec = isin(TEC['time'], dt)
        T0t = TEC['tecim'][idttec]
        T0x = T0t[:, idx, :]
        T0 = T0x[:, :, idy]
        tecdt = TEC['time'][idttec]
    
    # Save
    if platform.system() == 'Linux':
        odir = P.odir if P.odir is not None else '/media/smrak/gnss/images/'
        odir += dirdatetime + '_' + str(int(altkm)) + '_' + str(average) + '_' + str(clim[1]).replace(".", "")
        if nightshade:
            odir += '_ns'
        if P.tec is not None:
            odir += '_percent'
        odir += '/'
    elif platform.system() == 'Windows':
        odir = P.odir if P.odir is not None else os.path.split(gpsfn)[0] + '\\images\\'
        odir += dirdatetime + '_' + str(int(altkm)) + '_' + str(average) + '_' + str(clim[1]).replace(".", "")
        if nightshade:
            odir += '_ns'
        if P.tec is not None:
            odir += '_percent'
        odir += '\\'
    #RUN
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        im = [ex.submit(makeImage, squeeze(im[i : i+average]), fillpixel_iter) for i in iterate1]
    #
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
                          nightshade=nightshade, ns_alpha=0.05,
                          mlon_levels=mag_meridians, mlat_levels=mag_parallels,
                          mlon_labels=False, mlat_labels=False, mgrid_style='--',
                          mlon_colors='w', mlat_colors='w', terminator=1, terminator_altkm=350,
                          )
        image = im[j].result()
        j+=1
        # dTEC/TEC ?
        if fntec is not None:
            assert os.path.exists(fntec)
            idttec0 = abs(tecdt - dt[i]).argmin()
            assert abs(tecdt[idttec0] - dt[i]) < timedelta(minutes=10)
            tecim = T0[idttec0]
            T00 = interpolateTEC(im=tecim, x0=xgtec, y0=ygtec, 
                                 xgrid=xg, ygrid=yg, 
                                 method='linear')
            image = divide(image, T00) * 100
            label = 'dTEC [%]'
        else:
            label = 'dTEC [TECu]'

        # Plot image
        try:
            if image_type == 'contourf':
                levels = linspace(clim[0], clim[1], 40)
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
            fig.colorbar(imax, cax=cax, label=label, 
                         ticks=[clim[0], clim[0]/2, 0, clim[1]/2, clim[1]])

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

        if not os.path.exists(odir):
            import subprocess
            if platform.system() == 'Linux': 
                subprocess.call('mkdir -p {}'.format(odir), shell=True, timeout=2)
            elif platform.system() == 'Windows':
                subprocess.call('mkdir "{}"'.format(odir), shell=True, timeout=2)
        tit = dt[i].strftime('%m%d_%H%M')
        ofn = odir+str(tit)+'.png'
        plt.savefig(ofn, dpi=150)
        plt.close()
