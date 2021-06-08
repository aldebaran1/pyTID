# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:48:42 2020

@author: smrak@bu.edu
"""
import yaml, os, platform
import h5py
from datetime import datetime, timedelta
from dateutil import parser
import numpy as np
from gpstec import gpstec
from cartomap import geogmap as gm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pymap3d import aer2geodetic


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
             latlim=None, lonlim=None, res=None):
    """
    """
    xgrid, ygrid = np.meshgrid(np.arange(lonlim[0], lonlim[1]+.01, res),
                               np.arange(latlim[0], latlim[1]+.01, res))
    im = np.nan * np.copy(xgrid)
    # Fill out the image pixels
    for i in range(glon.size):
        idx = abs(xgrid[0, :] - glon[i]).argmin() if abs(xgrid[0, :] - glon[i]).min() < 3*res else np.nan
        idy = abs(ygrid[:, 0] - glat[i]).argmin() if abs(ygrid[:, 0] - glat[i]).min() < 3*res else np.nan
        # If image indexes are valid
        if np.isfinite(idx) and np.isfinite(idy):
            # Assign the value to the pixel
            if np.isnan(im[idy,idx]):
                im[idy,idx] = tid[i]
            # If this is not the first value to assign, assign a
            # mean of both values
            else:
                im[idy, idx] = np.nanmedian( [im[idy, idx], + tid[i]])
        im = fillPixels(im, 1)
    return xgrid, ygrid, im


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
    p.add_argument('--projection', type=str, default=None)
    p.add_argument('--cmap', type=str, default=None)
    p.add_argument('-s', '--size', help='scatter size', type=int, default=15)
    p.add_argument('--latlim', type=float, nargs=2, default=None)
    p.add_argument('--lonlim', type=float, nargs=2, default=None)
    p.add_argument('--tec', type=str, help='TEC file', default=None)
    p.add_argument('--scint', type=str, help='Scint file', default=None)
    p.add_argument('--image', action='store_true')
    p.add_argument('-r', '--resolution', default=0.5, type=float)

    P = p.parse_args()
    altkm = P.altkm
    clim = P.clim
    scatter_size = P.size
    terminator = P.terminator
    terminator_altkm = altkm if P.terminator_altkm is None else P.terminator_altkm
    
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
    if platform.system() == 'Linux':
        odir = P.odir if P.odir is not None else '/media/smrak/gnss/images/'
        if P.image:
            odir += dirdatetime + '_image_' + str(int(altkm)) + '_' + str(average) + '_' + cmap + '_' + str(clim[1]).replace('.', '') + '_' + str(abs(lonlim[0])) + '_' + str(abs(lonlim[1]))
        else:
            odir += dirdatetime + '_scatter_' + str(int(altkm)) + '_' + str(average) + '_' + cmap + '_' + str(clim[1]).replace('.', '') + '_' + str(abs(lonlim[0])) + '_' + str(abs(lonlim[1]))
        if P.tec is not None:
            odir += '_percent'
        odir += '/'
    elif platform.system() == 'Windows':
        odir = P.odir if P.odir is not None else os.path.split(gpsfn)[0] + '\\images\\'
        if P.image:
            odir += dirdatetime + '_image_' + str(int(altkm)) + '_' + str(average) + '_' + cmap + '_' + str(clim[1]).replace('.', '') + '_' + str(abs(lonlim[0])) + '_' + str(abs(lonlim[1]))
        else:
            odir += dirdatetime + '_scatter_' + str(int(altkm)) + '_' + str(average) + '_' + cmap + '_' + str(clim[1]).replace('.', '') + '_' + str(abs(lonlim[0])) + '_' + str(abs(lonlim[1]))
        if P.tec is not None:
            odir += '_percent'
        odir += '\\'
    
    j = 0
    for i in iterate:
        print ('Plotting figure {}/{}'.format(j+1, iterate.shape[0]))
        j += 1
        # Get a map
        fig, ax = gm.plotCartoMap(figsize=figure_size, projection=projection, #title=dt[i],
                          terrain=terrain, states=states, border_color=border_color,
                          background_color=background_color,
                          lonlim=lonlim,latlim=latlim,
                          title="{}, alt = {} km".format(dt[i], altkm),
                          meridians=meridians, parallels=parallels,
                          grid_linewidth=grid_linewidth,grid_color=grid_color,
                          apex=apex, mlon_cs=mlon_cs, date=dt[i],
                          mlon_levels=mag_meridians, mlat_levels=mag_parallels,
                          mlon_labels=False, mlat_labels=False, mgrid_style='--',
                          mlon_colors='w', mlat_colors='w', 
                          terminator=terminator, terminator_altkm=terminator_altkm,
                          )
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
            label = 'dTEC/TEC [%] '
        else:
            label = 'dTEC [TECu]'
#        try:
        # Convert coordinates
        TID = h5py.File(gpsfn, 'r')
        az = np.nanmedian(TID['az'][i-average:i+1, :, :], axis=0)
        el = np.nanmedian(TID['el'][i-average:i+1, :, :], axis=0)
        
        if average > i and i < dt.size:
            tid = np.nanmedian(TID['res'][i-average:i+1, :, :], axis=0).flatten()
        elif average < i:
            tid = np.nanmedian(TID['res'][i:i+2, :, :], axis=0).flatten()
        else:
            tid = np.nanmedian(TID['res'][i-average:i, :, :], axis=0).flatten()
        TID.close()
        
        r1 = (altkm * 1e3) / np.sin(np.radians(el))
        h0 = np.nan_to_num(rxp[:, 2])
        h0[h0 < 0] = 0
        ipp_lla = aer2geodetic(az=az, el=el, srange=r1, 
                               lat0=rxp[:,0], lon0=rxp[:,1], h0=h0)
        glon = ipp_lla[1].flatten()
        glat = ipp_lla[0].flatten()
        
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
            
            idf1 = (abs(tid) > 0.05) & (abs(tid) < clim[1]/4)
            idf2 = (abs(tid) >= clim[1]/4) & (abs(tid) < clim[1]/2)
            idf3 = abs(tid) > clim[1]/2
        else:
            idf1 = (abs(tid) >= 0.02) & (abs(tid) < clim[1]/4)
            idf2 = (abs(tid) >= clim[1]/4) & (abs(tid) < clim[1]/2)
            idf3 = abs(tid) > clim[1]/2
        # Z-axis
        if not P.image:
            imax = ax.scatter(glon[idf1], glat[idf1], c=tid[idf1], 
                               s = scatter_size, alpha=0.8, zorder=1,
                               cmap=cmap, 
                               vmin = clim[0], vmax = clim[1],
                               transform=ccrs.PlateCarree())
            imax = ax.scatter(glon[idf2], glat[idf2], c=tid[idf2], 
                               s = scatter_size, alpha=0.8, zorder=2,
                               cmap=cmap, 
                               vmin = clim[0], vmax = clim[1],
                               transform=ccrs.PlateCarree())
            imax = ax.scatter(glon[idf3], glat[idf3], c=tid[idf3], 
                               s = scatter_size, alpha=0.8, zorder=3,
                               cmap=cmap, 
                               vmin = clim[0], vmax = clim[1],
                               transform=ccrs.PlateCarree())
            posn = ax.get_position()
            cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
            fig.colorbar(imax, cax=cax, label=label)
        else:
            xg, yg, im = ImageNew(glon, glat, tid, 
                                  lonlim=lonlim, latlim=latlim, res=P.resolution)
            pcmim = ax.pcolormesh(xg, yg, im.T, cmap=cmap, 
                       vmin = clim[0], vmax = clim[1],
                       transform=ccrs.PlateCarree())
            posn = ax.get_position()
            cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
            fig.colorbar(pcmim, cax=cax, label=label)
#        except Exception as e:
#            print (e)
        
        if P.scint:
            scintdata = h5py.File(P.scint, 'r')
            idtscint0 = abs(scint_dt - dt[i]).argmin()
            idtscint = np.zeros(scint_dt.size, dtype=bool)
            d_offset = int(average * 30)
            if idtscint0 >= d_offset:
                idtscint[idtscint0-d_offset : idtscint0] = True
            else:
                idtscint[:idtscint0+1] = True
            scintglon = scintdata['data/ipp'][idtscint0, :, :, 1]
            scintglat = scintdata['data/ipp'][idtscint0, :, :, 0]
            snr4ix = np.nanmedian(scintdata['data/snr4'][idtscint, :, :], axis=0)
            sigma_tec_ix = np.nanmedian(scintdata['data/sigma_tec'][idtscint, :, :], axis=0)
            try:
                idf = np.isfinite(snr4ix)
                ax.scatter(scintglon[idf], scintglat[idf], c='r',
                               s = snr4ix[idf]*400, alpha=0.9, zorder=20,
                               marker='x',
                               transform=ccrs.PlateCarree())
                idf = np.isfinite(sigma_tec_ix)
                ax.scatter(scintglon[idf], scintglat[idf], c='cyan',# edgecolors='r', facecolors='none',
                               s = (sigma_tec_ix[idf]*500)**2, alpha=0.9, zorder=10,
                               marker='+',
                               transform=ccrs.PlateCarree())
                scintdata.close()
            except:
                scintdata.close()
                print ("Scint im ...")
                
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