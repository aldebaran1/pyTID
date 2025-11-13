# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:48:42 2020

@author: smrak@bu.edu
"""
import yaml, os, platform
import h5py
from datetime import datetime, timedelta, timezone
from dateutil import parser
import numpy as np
from scipy import ndimage
from gpstec import gpstec
import subprocess
try:
    from cartomap import geogmap as gm
except:
    pass
try:
    import matplotlib.pyplot as plt
except:
    pass
try:
    import cartopy.crs as ccrs
except:
    pass
from pyGnss import pyGnss #
from astropy.convolution import Gaussian2DKernel, convolve

import warnings
warnings.filterwarnings("ignore")

keys = {'poly': 'res', 'ra': 'res_ra', 'sg': 'res_sg', 'sg1': 'res_sg1', 'sg2': 'res_sg2', 'sg3': 'res_sg3',}

def getNeighbours(image,i,j,N=3):
    """
    Return an array of <=9 neighbour pixel of an image with a center at (i,j)
    """
    nbg = []
    m = int(np.floor(N/2))
    M = int(np.ceil(N/2))
    for k in np.arange(i-m, i+M):
        for l in np.arange(j-m, j+M):
            try:
                nbg.append(image[k,l])
            except:
                pass
    return np.array(nbg)

def fillPixels(im, N=1):
    """
    Fill in the dead pixels. If a dead pixel has a least 4 finite neighbour
    pixel, than replace the center pixel with a mean valuse of the neighbours
    """
    X = im.shape[0]-1
    Y = im.shape[1]-1
    imcopy = np.copy(im)
    for n in range(N):
        skip = int(np.floor((3+n)/2))
        starti = 0
        startj = 0
        forwardi = int(np.floor(0.7*X))
        backwardi = int(np.floor(0.3*X))
        if n%2 == 0:
            for i in np.arange(starti, forwardi, skip):
                for j in np.arange(startj, Y, skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix = np.where(np.isfinite(nbg))[0]
                            avg = np.mean(nbg[ix])
                            im[i,j] = avg
            for i in np.arange(X, backwardi, -skip):
                for j in np.arange(Y, 0, -skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix = np.where(np.isfinite(nbg))[0]
                            avg = np.mean(nbg[ix])
                            im[i,j] = avg
        else:
            for j in np.arange(startj, Y, skip):
                for i in np.arange(starti, forwardi, skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix = np.where(np.isfinite(nbg))[0]
                            avg = np.mean(nbg[ix])
                            im[i,j] = avg

            for j in np.arange(Y, 0, -skip):
                for i in np.arange(X, backwardi, -skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix = np.where(np.isfinite(nbg))[0]
                            avg = np.mean(nbg[ix])
                            im[i,j] = avg
    return im

def ImageNew(glon, glat, tid, 
             latlim=None, lonlim=None, res=None,
             filter_type='gaussian', sigma=2, filter_size=5):
    """
    """
    xgrid, ygrid = np.meshgrid(np.arange(lonlim[0], lonlim[1]+.01, res),
                               np.arange(latlim[0], latlim[1]+.01, res))
    im = np.empty(xgrid.shape, dtype=object)
    # Fill out the image pixels
    for i in range(glon.size):
        idx = abs(xgrid[0, :] - glon[i]).argmin() if abs(xgrid[0, :] - glon[i]).min() < 3*res else np.nan
        idy = abs(ygrid[:, 0] - glat[i]).argmin() if abs(ygrid[:, 0] - glat[i]).min() < 3*res else np.nan
        # If image indexes are valid
        if np.isfinite(idx) and np.isfinite(idy):
            # Assign the value to the pixel
            if im[idy,idx] is None:
                im[idy,idx] = [tid[i]]
            # If this is not the first value to assign, assign a
            # mean of both values
            else:
                im[idy,idx].append(tid[i])
    
    imout = np.nan * np.empty(xgrid.shape)
    for i in range(xgrid.shape[0]):
        for j in range(xgrid.shape[1]):
            if im[i,j] is not None:
                imout[i,j] = np.nanmedian(im[i,j])
    if filter_type == 'median':
        imout = fillPixels(imout)
        imout = ndimage.median_filter(imout, filter_size)
    elif filter_type == 'gaussian':
        kernel = Gaussian2DKernel(x_stddev=sigma, y_stddev=sigma, x_size=filter_size, y_size=filter_size)
        imout = convolve(imout, kernel)
        
        imout[:filter_size, :] = np.nan
        imout[:, :filter_size] = np.nan
        imout[-filter_size:, :] = np.nan
        imout[:, -filter_size:] = np.nan
    del im
    return xgrid, ygrid, imout


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
    p.add_argument('--terminator', action='store_true')
    p.add_argument('--terminator_altkm', type=float, default=None)
    p.add_argument('--clim', type=float, nargs=2, default=[0,1])
    p.add_argument('--altkm', type=float, help='Projection altitude in km', default=350)
    p.add_argument('--average', type=int, default=1)
    p.add_argument('--mode', type=str, help='What kind of tid output? "poly" (default), "ra", "sg"', default='poly')
    p.add_argument('--projection', type=str, default=None)
    p.add_argument('--cmap', type=str, default=None)
    p.add_argument('-s', '--size', help='scatter size', type=int, default=15)
    p.add_argument('--latlim', type=float, nargs=2, default=None)
    p.add_argument('--lonlim', type=float, nargs=2, default=None)
    p.add_argument('--tec', type=str, help='TEC file', default=None)
    p.add_argument('--scint', type=str, help='Scint file', default=None)
    p.add_argument('--filter', type=str, default='gaussian', help='Default is Gaussian')
    p.add_argument('--filter_size', type=int, default=5, help='Default is 5 pixels')
    p.add_argument('--filter_sigma', type=float, default=1, help='Default is 1 pixels')
    p.add_argument('--elmask', type=int, default=None, help='Default is None')
    p.add_argument('-r', '--resolution', default=0.5, type=float, help='Default is 0.5') 
    p.add_argument('--save', action='store_true', help='Save to file?')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    
    P = p.parse_args()
    altkm = P.altkm
    clim = P.clim
    scatter_size = P.size
    terminator = P.terminator
    terminator_altkm = altkm if P.terminator_altkm is None else P.terminator_altkm
    
    filter_type = P.filter
    filter_size = P.filter_size
    filter_sigma = P.filter_sigma
    
    mode = P.mode
    key = keys[mode]
    
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
    # TID Images
    TID = h5py.File(gpsfn, 'r')
    time = TID['obstimes'][:]
    dt = np.array([datetime.utcfromtimestamp(t) for t in time])
    rxp = TID['rx_positions'][:]
    TID.close()
#    gpsdata.close()
    
    # Time conversion and filtering
    dirdatetime = dt[0].strftime('%Y%m%d')
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
        idt = (dt >= timelim[0]) & (dt <= timelim[1])
    else:
        idt = np.ones(dt.size, dtype=bool)
        
    iterate = np.arange(np.where(idt==1)[0][0], np.where(idt==1)[0][-1]+1, skip)
    
    if fntec is not None:
        assert os.path.exists(fntec)
        TEC = gpstec.readFromHDF(fntec)
        tecdt = TEC['time']
        tecim = TEC['tecim']
        xgrid = TEC['xgrid']
        ygrid = TEC['ygrid']
        tec_average = 3
    if P.scint is not None:
        assert os.path.exists(P.scint)
        scintdata = h5py.File(P.scint, 'r')
        scint_time = scintdata['data/time'][:]
        scint_dt = np.array([datetime.utcfromtimestamp(t) for t in scint_time])
        scintdata.close()
    # Save
    # if platform.system() in ('Linux', 'Darwin'):
    odir = P.odir if P.odir is not None else os.path.split(gpsfn)[0] + os.sep + 'images' + os.sep
    odir += f'{dirdatetime}_image_{altkm}_{key}_{P.resolution}_{average}_elmask{int(P.elmask)}_{cmap}_{filter_type}_{clim[0]}_{clim[1]}_{abs(lonlim[0])}_{abs(lonlim[1])}'
    if P.tec is not None:
        odir += '_percent'
    odir += os.sep
    
    j = 0
    for i in iterate:
        if P.verbose:
            print ('Plotting figure {}/{}'.format(j+1, iterate.shape[0]))
        j += 1
        TID = h5py.File(gpsfn, 'r')
        # Get a map
        if key == 'poly':
            title = f"{dt[i]}, alt = {altkm} km, res={P.resolution}, N={filter_size}, $\sigma_N$={filter_sigma}"
        else:
            if mode[-1] in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'):
                window = TID.attrs[f'window_size{mode[-1]}']
            else:
                window = TID.attrs['window_size']
            title = f"{dt[i]}, alt = {altkm} km, window = {mode}-{window} min, res={P.resolution}, N={filter_size}, $\sigma_N$={filter_sigma}"
        
        if fntec is not None:
            assert os.path.exists(fntec)
            itec0 = abs(tecdt - dt[i]).argmin()
            assert abs(tecdt[itec0] - dt[i]) < timedelta(minutes=10)
            if tec_average > itec0 and itec0 < tecdt.size:
                tec = np.nanmedian(tecim[itec0 - tec_average:i+1, :, :], axis=0)
            elif tec_average < i:
                tec = np.nanmedian(tecim[itec0:itec0+2, :, :], axis=0)
            else:
                tec = np.nanmedian(tecim[itec0-tec_average:i, :, :], axis=0)
            label = 'dTEC/TEC [%]'
        else:
            label = 'dTEC [TECu]'
        # Convert coordinates
        az = np.nanmedian(TID['az'][i-average:i+1, :, :], axis=0)
        el = np.nanmedian(TID['el'][i-average:i+1, :, :], axis=0)
        if P.elmask is not None:
            az[el<P.elmask] = np.nan
            el[el<P.elmask] = np.nan
        if average > i and i < dt.size:
            tid = np.nanmedian(TID[key][i-average:i+1, :, :], axis=0).flatten()
        elif average < i:
            tid = np.nanmedian(TID[key][i:i+2, :, :], axis=0).flatten()
        else:
            tid = np.nanmedian(TID[key][i-average:i, :, :], axis=0).flatten()
        TID.close()
        
        ipp_lat, ipp_lon = pyGnss.aer2ipp(az, el, rxp, H=altkm)
        glon = ipp_lon.flatten()
        glat = ipp_lat.flatten()
        
        idf = np.isfinite(glon) & np.isfinite(glat)
        glon = glon[idf]
        glat = glat[idf]
        tid = tid[idf]
        if fntec is not None:
            for k in range(glon.size):
                idx = abs(glon[k] - xgrid).argmin()
                idy = abs(glat[k] - ygrid).argmin()
                tid[k] = tid[k] / tec[idx, idy] if np.isfinite(tec[idx, idy]) else np.nan
            tid *= 100
        # Z-axis
        xg, yg, im = ImageNew(glon, glat, tid, lonlim=lonlim, latlim=latlim, 
                      res=P.resolution, filter_type=filter_type, sigma=filter_sigma,
                      filter_size=filter_size)
        
        if not os.path.exists(odir):
            if platform.system() in ('Linux', 'Darwin'): 
                subprocess.call('mkdir -p {}'.format(odir), shell=True, timeout=2)
            elif platform.system() == 'Windows':
                subprocess.call('mkdir "{}"'.format(odir), shell=True, timeout=2)
        
        if P.save:
            
            xfile = os.path.split(odir)[0] + os.sep + f'grid_{mode}_{window}min_' + \
                dt[i].strftime("%Y%m%d%H%M%S") + '_' + os.path.split(gpsfn)[1]
            X = h5py.File(xfile, 'w')
            X.create_dataset('time', data=dt[i].replace(tzinfo=timezone.utc).timestamp())
            X.create_dataset('glon', data=xg, compression="gzip",chunks=True, compression_opts=9)
            X.create_dataset('glat', data=yg, compression="gzip",chunks=True, compression_opts=9)
            X.create_dataset('tid', data=im, compression="gzip",chunks=True, compression_opts=9)
            X.attrs[u'resolution'] = P.resolution
            X.attrs[u'filter_type'] = filter_type
            X.attrs[u'filter_size'] = filter_size
            X.attrs[u'filter_sigma'] = filter_sigma
            X.close()
            
        
        else:
            fig, ax = gm.plotCartoMap(figsize=figure_size, projection=projection, #title=dt[i],
                              terrain=terrain, states=states, border_color=border_color,
                              background_color=background_color,
                              lonlim=lonlim,latlim=latlim,
                              title=title,
                              meridians=meridians, parallels=parallels,
                              grid_linewidth=grid_linewidth,grid_color=grid_color,
                              apex=apex, mlon_cs=mlon_cs, date=dt[i],
                              mlon_levels=mag_meridians, mlat_levels=mag_parallels,
                              mlon_labels=False, mlat_labels=False, mgrid_style='--',
                              mlon_colors='w', mlat_colors='w', 
                              terminator=terminator, terminator_altkm=terminator_altkm,
                              )
            pcmim = ax.pcolormesh(xg, yg, im, cmap=cmap, 
                       vmin = clim[0], vmax = clim[1],
                       transform=ccrs.PlateCarree())
            posn = ax.get_position()
            cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
            fig.colorbar(pcmim, cax=cax, label=label)
            
            tit = dt[i].strftime('%m%d_%H%M')
            ofn = odir+str(tit)+'.png'
            plt.savefig(ofn, dpi=50)
            plt.close()
        
        