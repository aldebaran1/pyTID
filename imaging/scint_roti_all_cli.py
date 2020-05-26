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
        roti = scintdata['data/roti'][scint_idt, :, :]
        S4 = scintdata['data/s4'][scint_idt, :, :]
        # Plot
        fig = plt.figure(figsize=[14,10])
        ax0 = plt.subplot(221, projection=ccrs.Stereographic(central_longitude=(sum(lonlim)/2)))
        ax1 = plt.subplot(222, projection=ccrs.Stereographic(central_longitude=(sum(lonlim)/2)))
        ax2 = plt.subplot(223, projection=ccrs.Stereographic(central_longitude=(sum(lonlim)/2)))
        ax3 = plt.subplot(224, projection=ccrs.Stereographic(central_longitude=(sum(lonlim)/2)))
    
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
        ax2 = cm.plotCartoMap(latlim=latlim, lonlim=lonlim, projection='stereo',
                              meridians=None, parallels=None, ax=ax2,
                              grid_linewidth=1, states = False,
                              title=it, background_color='grey',
                              apex=True, mlat_levels=[-20,0,20,40,60,80,90],
                              mlat_colors='w', mgrid_width=1, mgrid_style='--',
                              mlon_levels=np.arange(0,361,40), mlat_labels=False,
                              mlon_colors='w', mlon_labels=False)
        ax3 = cm.plotCartoMap(latlim=latlim, lonlim=lonlim, projection='stereo',
                              meridians=None, parallels=None, ax=ax3,
                              grid_linewidth=1, states = False,
                              title=it, background_color='grey',
                              apex=True, mlat_levels=[-20,0,20,40,60,80,90],
                              mlat_colors='w', mgrid_width=1, mgrid_style='--',
                              mlon_levels=np.arange(0,361,40), mlat_labels=False,
                              mlon_colors='w', mlon_labels=False)
        # ------------------------------------------------------------------------- - #
        ax0.pcolormesh(xgrid, ygrid, tecim.T, cmap='gray', 
                             vmin = tecclim[0], vmax = tecclim[1], 
                            transform=ccrs.PlateCarree())
        im1 = ax1.pcolormesh(xgrid, ygrid, tecim.T, cmap='gray', #'nipy_spectral'
                             vmin = tecclim[0], vmax = tecclim[1], 
                            transform=ccrs.PlateCarree())
        ax2.pcolormesh(xgrid, ygrid, tecim.T, cmap='gray', 
                             vmin = tecclim[0], vmax = tecclim[1], 
                            transform=ccrs.PlateCarree())
        ax3.pcolormesh(xgrid, ygrid, tecim.T, cmap='gray', 
                             vmin = tecclim[0], vmax = tecclim[1], 
                            transform=ccrs.PlateCarree())
        # Scint with amplitude
        if np.sum(np.isfinite(sigma_tec)) > 0:
            imst = ax0.scatter(ipp_lon, ipp_lat,
                        c = sigma_tec,
                        s = 30, #(sigma_tec)**2 * 1000000,
                        marker = 'o',
                        cmap='Reds',
                        alpha=0.8,
                        vmin=0, vmax=0.08,
    #                    facecolors = 'none',
                        transform = ccrs.PlateCarree())
        imsnr4 = ax1.scatter(ipp_lon, ipp_lat,
                    c = snr4,
                    s = 30, #np.square(snr4) * 1000,
                    marker = 'o',
                    alpha = 0.8,
                    cmap='Blues',
                    vmin=0,vmax=0.3,
                    transform = ccrs.PlateCarree())
        # Scint locations
        imroti = ax2.scatter(ipp_lon, ipp_lat,
                        c = roti,
                        s = 30,
                        marker = 'o',
                        alpha=0.8,
                        vmin=0,vmax=0.02,
                        cmap='Reds',
                        transform = ccrs.PlateCarree())
        ims4 = ax3.scatter(ipp_lon, ipp_lat,
                        c = S4,
                        s = 30,
                        marker = 'o',
                        alpha=0.8,
                        vmin=0,vmax=0.2,
                        cmap='Blues',
                        transform = ccrs.PlateCarree())
        posn0 = ax0.get_position()
        cax = fig.add_axes([posn0.x0+posn0.width+0.01, posn0.y0, 0.02, posn0.height])
        fig.colorbar(imst, cax=cax, label='$\sigma_{TEC}$ [TECu]')
        
        posn1 = ax1.get_position()
        cax = fig.add_axes([posn1.x0+posn1.width+0.01, posn1.y0, 0.02, posn1.height])
        fig.colorbar(imsnr4, cax=cax, label='$SNR_4$')
        
        posn2 = ax2.get_position()
        cax = fig.add_axes([posn2.x0+posn2.width+0.01, posn2.y0, 0.02, posn2.height])
        fig.colorbar(imroti, cax=cax, label='$ROTI$')
        
        posn3 = ax3.get_position()
        cax = fig.add_axes([posn3.x0+posn3.width+0.01, posn3.y0, 0.02, posn3.height])
        fig.colorbar(ims4, cax=cax, label='S4')
        cax = fig.add_axes([posn3.x0, posn3.y0-0.03, posn3.width, 0.02])
        fig.colorbar(im1, cax=cax, label='TEC [TECu]', orientation='horizontal')
        
        if SAVE:
#            plt.tight_layout()
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
