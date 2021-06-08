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
    p.add_argument('--clim', type=float, nargs=2, default=[0,1])
    p.add_argument('--altkm', type=float, help='Projection altitude in km', default=350)
    p.add_argument('--average', type=int, default=1)
    p.add_argument('--projection', type=str, default=None)
    p.add_argument('--cmap', type=str, default=None)
    p.add_argument('-s', '--size', help='scatter size', type=int, default=15)
    p.add_argument('--latlim', type=float, nargs=2, default=None)
    p.add_argument('--lonlim', type=float, nargs=2, default=None)
    p.add_argument('--tec', type=str, help='TEC file', default=None)
    p.add_argument('--terminator', action='store_true')
    p.add_argument('--terminator_altkm', type=float, default=None)

    P = p.parse_args()
    altkm = P.altkm
    clim = P.clim
    scatter_size = P.size
    terminator= P.terminator
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
    # GPS Images
    gpsdata = h5py.File(gpsfn, 'r')
    time = gpsdata['obstimes'][:]
    rxp = gpsdata['rx_positions'][:]
    az = gpsdata['az'][:]
    el = gpsdata['el'][:]
    data = gpsdata['roti'][:]
    # compute IPP
    gpsdata.close()
    
    # Time conversion and filtering
    dt = np.array([datetime.utcfromtimestamp(t) for t in time])
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
        idttec = (TEC['time'] >= timelim[0]) & (TEC['time'] <= timelim[1])
        tecdt = TEC['time'][idttec]
        tecim = TEC['tecim'][idttec]
        xgrid = TEC['xgrid']
        ygrid = TEC['ygrid']
    
    # Save
    if platform.system() == 'Linux':
        odir = P.odir if P.odir is not None else '/media/smrak/gnss/roti/'
        odir += dirdatetime + '_' + str(int(altkm)) + '_' + str(average) + '_' + cmap + '_' + str(clim[1]).replace('.', '')
        odir += '/'
    elif platform.system() == 'Windows':
        odir = P.odir if P.odir is not None else os.path.split(gpsfn)[0] + '\\roti\\'
        odir += dirdatetime + '_' + str(int(altkm)) + '_' + str(average) + '_' + cmap + '_' + str(clim[1]).replace('.', '')
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
        # dTEC/TEC ?
        if fntec is not None:
            assert os.path.exists(fntec)
            idttec0 = abs(tecdt - dt[i]).argmin()
            assert abs(tecdt[idttec0] - dt[i]) < timedelta(minutes=10)
            tecax = ax.pcolormesh(xgrid, ygrid, tecim[idttec0].T, cmap='gray',
                          vmin = 0, vmax = 40,
                          transform=ccrs.PlateCarree())
            posn = ax.get_position()
            cax = fig.add_axes([posn.x0, posn.y0-0.03, posn.width, 0.02])
            fig.colorbar(tecax, cax=cax, label='TEC [TECu]', orientation='horizontal')

        # Plot image
        r1 = (altkm * 1e3) / np.sin(np.radians(el[i]))
        h0 = np.nan_to_num(rxp[:, 2])
        h0[h0 < 0] = 0
        ipp_lla = aer2geodetic(az=az[i], el=el[i], srange=r1, 
                               lat0=rxp[:,0], lon0=rxp[:,1], h0=h0)
        glon = ipp_lla[1]
        glat = ipp_lla[0]
        try:
            # Convert coordinates
            
            idf = np.isfinite(glon) & np.isfinite(glat)
            if average > i and i < dt.size:
                rotia = np.nanmedian(data[i-average:i+1], axis=0) * 60
            elif average < i:
                rotia = np.nanmedian(data[i:i+2], axis=0) * 60
            else:
                rotia = np.nanmedian(data[i-average:i], axis=0) * 60
            # Z-axis
            idf1 = rotia[idf] < 0.3
            idf2 = (rotia[idf] >= 0.3) & (rotia[idf] < 1)
            idf3 = rotia[idf] > 1
            imax = ax.scatter(glon[idf][idf1], glat[idf][idf1], c=rotia[idf][idf1], 
                               s = scatter_size, alpha=0.9, zorder=1,
                               cmap=cmap, 
                               vmin = clim[0], vmax = clim[1],
                               transform=ccrs.PlateCarree())
            imax = ax.scatter(glon[idf][idf2], glat[idf][idf2], c=rotia[idf][idf2], 
                               s = scatter_size, alpha=0.9, zorder=2,
                               cmap=cmap, 
                               vmin = clim[0], vmax = clim[1],
                               transform=ccrs.PlateCarree())
            imax = ax.scatter(glon[idf][idf3], glat[idf][idf3], c=rotia[idf][idf3], 
                               s = scatter_size, alpha=0.9, zorder=3,
                               cmap=cmap, 
                               vmin = clim[0], vmax = clim[1],
                               transform=ccrs.PlateCarree())
            posn = ax.get_position()
            cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
            fig.colorbar(imax, cax=cax, label='ROTI [TECu/min]')

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