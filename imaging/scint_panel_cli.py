#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:02:09 2019

@author: smrak
"""
from gpstec import gpstec
import h5py
import os, glob
from datetime import datetime, timedelta
from dateutil import parser
import numpy as np
import cartopy.crs as ccrs
import cartomap.geogmap as cm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import platform

def main(date='', root=None, scintfn=None, 
         clim=None, tlim=None, trange=2.5, 
         odir=None,
         SAVE=1, resolution=5):
    if tlim is None:
        tlim = [parser.parse(date), parser.parse(date) + timedelta(days=1)]
    if isinstance(tlim[0], str):
        tlim = [parser.parse(tlim[0]), parser.parse(tlim[1])]
    assert isinstance(tlim[0], datetime) and isinstance(tlim[1], datetime)
    obstimes = []
    t = tlim[0]
    while t <= tlim[1]:
        obstimes.append(t)
        t += timedelta(minutes=resolution)
    
    latlim=[-10, 75]
    lonlim=[-160, -50]
    DPI = 100
    if clim is None:
        tecclim = [0, 20]
    else:
        tecclim = clim
    if root is None:
        if platform.system() == 'Windows':
            root = os.getcwd()
        else:
            root = '/home/smrak/Documents/scintillation/'
    if odir is None:
        if platform.system() == 'Windows':
            odir = root + '\\maps\\{}\\{}\\bw-{}-{}\\'.format(tlim[0].year, parser.parse(date).strftime("%Y%m%d"), tecclim[0], tecclim[1])
            TECFN = root + '\\maps\\{}\\{}\\conv_{}T0000-{}T0000.h5'.format(tlim[0].year, parser.parse(date).strftime("%Y%m%d"), 
                                                                           tlim[0].strftime("%m%d"), tlim[1].strftime("%m%d"))
        else:
            odir =  root + '/maps/{}/bw-{}-{}/'.format(parser.parse(date).strftime("%Y%m%d"), tecclim[0], tecclim[1])
            TECFN = '/media/smrak/figures/gpstec/{}/{}/conv_{}T0000-{}T0000.h5'.format(parser.parse(date).year, 
                                                 parser.parse(date).strftime("%m%d"),
                                                 tlim[0].strftime("%Y%m%d"), tlim[1].strftime("%Y%m%d"))
    if platform.system() == 'Windows':
        TECFN = root + '\\maps\\{}\\{}\\conv_{}T0000-{}T0000.h5'.format(tlim[0].year, parser.parse(date).strftime("%Y%m%d"), 
                                                                           tlim[0].strftime("%m%d"), (tlim[0]+timedelta(days=1)).strftime("%m%d"))
    else:
        TECFN = '/media/smrak/figures/gpstec/{}/{}/conv_{}T0000-{}T0000.h5'.format(parser.parse(date).year, 
                                                 parser.parse(date).strftime("%m%d"),
                                                 tlim[0].strftime("%Y%m%d"), (tlim[0]+timedelta(days=1)).strftime("%Y%m%d"))
    assert os.path.isfile(TECFN), TECFN
    if scintfn is None:
        if platform.system() == 'Windows':
            scint_root = root + '\\hdf\\{}\\'.format(tlim[0].year)
            
        else:
            scint_root = root + '/hdf/'
        scint_fn_list = sorted(glob.glob(scint_root + "ix_{}_{}T*.h5".format(tlim[0].year, tlim[0].strftime("%m%d"))))
        assert len(scint_fn_list) > 0
        scintfn = scint_fn_list[0]
    assert os.path.isfile(scintfn)

# --------------------------------------------------------------------------- #
    for ii, it in enumerate(obstimes):
        # TEC data
        D = gpstec.readFromHDF(TECFN)
        tectime = D['time']
        xgrid = D['xgrid']
        ygrid = D['ygrid']
        idt_tec = abs(tectime - it).argmin()
        tecim = D['tecim'][idt_tec]
        # Scintillation data
        scintdata = h5py.File(scintfn, 'r')
        scint_time = scintdata['data/time'][:]
        scint_dt = np.array([datetime.utcfromtimestamp(t) for t in scint_time])
        # Filter out time range of interest
        scint_idt = np.zeros(scint_dt.size, dtype=bool)
        time_range = np.where( (scint_dt >= it-timedelta(minutes=trange)) & (scint_dt <= it+timedelta(minutes=trange)) )[0]
        scint_idt[time_range[0]:time_range[-1]+1] = True
    #    scint_idt[time_range[0]] = True
        # Read in data
        ipp_lat = scintdata['data/ipp'][scint_idt, :, :, 0]
        ipp_lon = scintdata['data/ipp'][scint_idt, :, :, 1]
        sigma_tec = scintdata['data/sigma_tec'][scint_idt, :, :]
        snr4 = scintdata['data/snr4'][scint_idt, :, :]
        # Plot
        fig = plt.figure(figsize=[12,6])
        ax0 = plt.subplot(121, projection=ccrs.Stereographic(central_longitude=(sum(lonlim)/2)))
        ax1 = plt.subplot(122, projection=ccrs.Stereographic(central_longitude=(sum(lonlim)/2)))
    
        ax0 = cm.plotCartoMap(latlim=latlim, lonlim=lonlim, projection='stereo',
                              meridians=None, parallels=None, ax=ax0,
                              grid_linewidth=1, states = False,
                              title=it, background_color='grey',
                              apex=True, mlat_levels=[-20,0,20,40,60,80,90],
                              mlat_colors='w', mgrid_width=1, mgrid_style='--',
                              mlon_levels=np.arange(0,361,40), mlat_labels=False,
                              mlon_colors='w', mlon_labels=False)
        
        ax1 = cm.plotCartoMap(latlim=latlim, lonlim=lonlim, projection='stereo',
                              meridians=None, parallels=None, ax=ax1,
                              grid_linewidth=1, states = False,
                              title=it, background_color='grey',
                              apex=True, mlat_levels=[-20,0,20,40,60,80,90],
                              mlat_colors='w', mgrid_width=1, mgrid_style='--',
                              mlon_levels=np.arange(0,361,40), mlat_labels=False,
                              mlon_colors='w', mlon_labels=False)
        # ------------------------------------------------------------------------- - #
        im0 = ax0.pcolormesh(xgrid, ygrid, tecim.T, cmap='gray', 
                             vmin = tecclim[0], vmax = tecclim[1], 
                            transform=ccrs.PlateCarree())
        im1 = ax1.pcolormesh(xgrid, ygrid, tecim.T, cmap='gray', #'nipy_spectral'
                             vmin = tecclim[0], vmax = tecclim[1], 
                            transform=ccrs.PlateCarree())
        # Scint with amplitude
        if np.sum(np.isfinite(sigma_tec)) > 0:
            ax0.scatter(ipp_lon, ipp_lat,
                        c = 'r',
                        s = (sigma_tec)**2 * 1000000,
                        marker = '+',
                        alpha=0.2,
    #                    facecolors = 'none',
                        transform = ccrs.PlateCarree())
        if np.sum(np.isfinite(snr4)) > 0:
            ax0.scatter(ipp_lon, ipp_lat,
                        c = 'b',
                        s = np.square(snr4) * 1000,
                        linewidth = 0.8,
                        marker = 'x',
                        alpha = 1,
    #                    facecolors = 'none',
                        transform = ccrs.PlateCarree())
        # Scint locations
        if np.sum(np.isfinite(sigma_tec)) > 0:
            maskst = np.isfinite(sigma_tec)
            ax1.scatter(ipp_lon[maskst], ipp_lat[maskst],
                        c = 'r',
                        s = 2,
                        marker = '.',
                        alpha=0.5,
                        transform = ccrs.PlateCarree())
        if np.sum(np.isfinite(snr4)) > 0:
            masks4 = np.isfinite(snr4)
            ax1.scatter(ipp_lon[masks4], ipp_lat[masks4],
                        c = 'b',
                        s = 10,
                        lw = 0.2,
                        marker = 'x',
                        alpha = 0.5,
                        transform = ccrs.PlateCarree())
            
        posn = ax1.get_position()
        cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
        fig.colorbar(im1, cax=cax, label='TEC [TECu]')
        
        if SAVE:
            if not os.path.exists(odir):
                import subprocess
                if platform.system() == 'Linux':
                    subprocess.call('mkdir -p "{}"'.format(odir), shell=True, timeout=2)
                elif platform.system() == 'Windows':
                    subprocess.call('mkdir "{}"'.format(odir), shell=True, timeout=2)
            print ("Plotting {}/{} - {}".format(ii+1, len(obstimes), it))
            fig.savefig(odir+'{}.png'.format(it.strftime('%m%d_%H%M')), dpi=DPI)
            plt.close(fig)
            
            del fig
            del sigma_tec
            del snr4

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('date')
    p.add_argument('--infn', type = str, help = 'Insert specific input scint-hdf5 file', default=None)
    p.add_argument('--clim', type = int, default = None, help="Colorbar limits for TEC", nargs=2)
    p.add_argument('--tlim', type = str, default = None, help="time limits", nargs=2)
    p.add_argument('--odir', type = str, default = None, help="Path to output files")
    p.add_argument('-r', '--resolution',  type = int, default = 5, help="Time resolution, defult 5min")
    P = p.parse_args()
    
    main(date = P.date, scintfn=P.infn, clim=P.clim, tlim=P.tlim, 
         resolution=P.resolution, odir=P.odir)
