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
from sunrise import terminator as ter
from apexpy import Apex

A = Apex()

def main(date='', root=None, scintfn=None, 
         clim=None, tlim=None, trange=None, 
         odir=None, latlim = None, lonlim = None,
         terminator=False, nightshade=False,
         cmap='gray',
         SAVE=1, resolution=5):
    DPI = 150
    
    datedt = [parser.parse(date), parser.parse(date) + timedelta(days=1)]
    
    if trange is None:
        trange = resolution / 2
    if latlim is None:
        latlim=[-10, 75]
    if lonlim is None:
        lonlim=[-160, -50]
    if clim is None:
        tecclim = [0, 20]
    else:
        tecclim = clim
    if root is None:
        if platform.system() == 'Windows':
            root = 'G:\\My Drive\\scintillation_data\\'
        else:
            root = '/home/smrak/Documents/scintillation/'
    if odir is None:
        if platform.system() == 'Windows':
            odir = root + '{}\\{}-{}-{}\\'.format(parser.parse(date).strftime("%Y%m%d"), 
                           cmap, tecclim[0], tecclim[1])
        else:
            odir =  root + '/maps/{}/{}-{}-{}/'.format(parser.parse(date).strftime("%Y%m%d"), 
                                  cmap, tecclim[0], tecclim[1])

    if platform.system() == 'Windows':
        TECFN = root + 'tid\\{}\\conv_{}T0000-{}T0000.h5'.format(parser.parse(date).strftime("%Y%m%d"), 
                                                                 datedt[0].strftime("%Y%m%d"), datedt[1].strftime("%Y%m%d"))
    else:
        TECFN = '/media/smrak/figures/gpstec/{}/{}/conv_{}T0000-{}T0000.h5'.format(parser.parse(date).year, 
                                                 parser.parse(date).strftime("%m%d"),
                                                 datedt[0].strftime("%Y%m%d"), datedt[1].strftime("%Y%m%d"))
    assert os.path.isfile(TECFN), TECFN
    if scintfn is None:
        if platform.system() == 'Windows':
            scint_root = root + '\\hdf\\{}\\'.format(datedt[0].year)
            
        else:
            scint_root = root + '/hdf/'
        scint_fn_list = sorted(glob.glob(scint_root + "ix_{}_{}T*.h5".format(datedt[0].year, datedt[0].strftime("%m%d"))))
        assert len(scint_fn_list) > 0
        scintfn = scint_fn_list[0]
    assert os.path.isfile(scintfn)
    
    #TEC
    TEC = gpstec.readFromHDF(TECFN)
    tectime = TEC['time']
    xgrid = TEC['xgrid']
    ygrid = TEC['ygrid']
    #SCINT DATA
    scintdata = h5py.File(scintfn, 'r')
    scint_time = scintdata['data/time'][:]
    scint_dt = np.array([datetime.utcfromtimestamp(t) for t in scint_time])
    
    if tlim is None:
        tlim = [parser.parse(date), parser.parse(date) + timedelta(days=1)]
    if isinstance(tlim[0], str):
        dirnametime = scint_dt[0].strftime('%Y%m%d')
        if dirnametime != parser.parse(P.tlim[0]).strftime('%Y%m%d'):
            t0 = parser.parse(dirnametime + 'T' + P.tlim[0])
            t1 = parser.parse(dirnametime + 'T' + P.tlim[1])
        else:
            t0 = parser.parse(P.tlim[0])
            t1 = parser.parse(P.tlim[0])
        tlim = [t0, t1]
    assert isinstance(tlim[0], datetime) and isinstance(tlim[1], datetime)
    obstimes = []
    t = tlim[0]
    while t <= tlim[1]:
        obstimes.append(t)
        t += timedelta(minutes=resolution)
    
# --------------------------------------------------------------------------- #
    for ii, it in enumerate(obstimes):
        # TEC data
        idt_tec = abs(tectime - it).argmin()
        if idt_tec < tectime.size-2:
            tecim = np.nanmean(TEC['tecim'][idt_tec:idt_tec+2], axis=0)
        else:
            tecim = TEC['tecim'][idt_tec]
        # Scintillation data
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
        # Plot
        fig = plt.figure(figsize=[15,6])
        ax0 = plt.subplot(121, projection=ccrs.Stereographic(central_longitude=(sum(lonlim)/2)))
        ax1 = plt.subplot(122, projection=ccrs.Stereographic(central_longitude=(sum(lonlim)/2)))
    
        ax0 = cm.plotCartoMap(latlim=latlim, lonlim=lonlim, projection='stereo',
                              meridians=None, parallels=None, ax=ax0,
                              grid_linewidth=1, states = False,
                              title=it, background_color='grey',
                              apex=True, mlat_levels=[-40,-20,0,20,40,60,80,90],
                              mlat_colors='w', mgrid_width=1, mgrid_style='--',
                              nightshade=nightshade, terminator=terminator,
                              terminator_altkm=350, ter_color='r', ter_style='-',
                              ter_width=2, mlon_cs='mlt', date=it,
                              mlon_levels=np.arange(0,24.1,4), mlat_labels=False,
                              mlon_colors='w', mlon_labels=False)
        
        ax1 = cm.plotCartoMap(latlim=latlim, lonlim=lonlim, projection='stereo',
                              meridians=None, parallels=None, ax=ax1,
                              grid_linewidth=1, states = False,
                              title=it, background_color='grey',
                              apex=True, mlat_levels=[-40,-20,0,20,40,60,80,90],
                              nightshade=nightshade, terminator=terminator,
                              terminator_altkm=350, ter_color='r', ter_style='-',
                              ter_width=2, mlon_cs='mlt', date=it,
                              mlat_colors='w', mgrid_width=1, mgrid_style='--',
                              mlon_levels=np.arange(0,24.1,4), mlat_labels=False,
                              mlon_colors='w', mlon_labels=False)
        
        glon_ter, glat_ter = ter.get_terminator(it, alt_km = 350)
        idlon = (glon_ter  > -160) & (glon_ter  < 0)
        mlat_ter, mlon_ter = A.convert(glat_ter[idlon], glon_ter[idlon], 'geo', 'apex', height=350)
        mlat_south = (mlat_ter < 0)
        glat_ter_conj, glon_ter_conj = A.convert(-mlat_ter[mlat_south], mlon_ter[mlat_south], 'apex', 'geo', height=350)
        ax0.plot(np.unwrap(glon_ter_conj,180), glat_ter_conj, 
                 '--r', lw=2, transform=ccrs.PlateCarree())
        # ------------------------------------------------------------------------- - #
        ax0.pcolormesh(xgrid, ygrid, tecim.T, cmap=cmap, 
                       vmin = tecclim[0], vmax = tecclim[1], 
                       transform=ccrs.PlateCarree())
        im1 = ax1.pcolormesh(xgrid, ygrid, tecim.T, cmap=cmap, #'nipy_spectral'
                             vmin = tecclim[0], vmax = tecclim[1], 
                            transform=ccrs.PlateCarree())
        # Scint with amplitude
#        if np.sum(np.isfinite(sigma_tec)) > 0:
        if trange >= 1:
            idf0 = np.isfinite(sigma_tec)
            if np.sum(np.isfinite(idf0)) == 0:
                idf0[0]=True
            imst = ax0.scatter(ipp_lon[idf0], ipp_lat[idf0],
                            c = sigma_tec[idf0],
                            s = 30, #(sigma_tec)**2 * 1000000,
                            marker = 'o',
                            cmap='Reds',
                            alpha=0.8,
                            vmin=0, vmax=0.05,
        #                    facecolors = 'none',
                            transform = ccrs.PlateCarree())
            idf0 = np.isfinite(snr4)
            if np.sum(np.isfinite(idf0)) == 0:
                idf0[0]=True
    #        if np.sum(np.isfinite(snr4)) > 0:
            imsnr4 = ax0.scatter(ipp_lon[idf0], ipp_lat[idf0],
                        c = snr4[idf0],
                        s = 30, #np.square(snr4) * 1000,
    #                    linewidth = 0.8,
                        marker = 'o',
                        alpha = 0.8,
                        cmap='Blues',
                        vmin=0, vmax=1.2,
    #                    facecolors = 'none',
                        transform = ccrs.PlateCarree())
            # Scint locations
            if np.sum(np.isfinite(roti)) > 0:
                idf0 = np.isfinite(roti)
                imroti = ax1.scatter(ipp_lon[idf0], ipp_lat[idf0],
                            c = roti[idf0],
                            s = 15,
                            marker = 'o',
                            alpha=0.8,
                            vmin=0,vmax=0.02,
                            cmap='jet',
                            transform = ccrs.PlateCarree())
            
            posn0 = ax0.get_position()
            cax = fig.add_axes([posn0.x0, posn0.y0-0.03, posn0.width, 0.02])
            fig.colorbar(imsnr4, cax=cax, label='$SNR_4$', orientation='horizontal')
            posn1 = ax1.get_position()
            cax = fig.add_axes([posn1.x0, posn1.y0-0.03, posn1.width, 0.02])
            fig.colorbar(imroti, cax=cax, label='ROTI [TECu]', orientation='horizontal')
            cax = fig.add_axes([posn1.x0+posn1.width+0.01, posn1.y0, 0.02, posn1.height])
            fig.colorbar(im1, cax=cax, label='TEC [TECu]')
            posn0 = ax0.get_position()
            cax = fig.add_axes([posn0.x0+posn0.width+0.01, posn0.y0, 0.02, posn0.height])
            fig.colorbar(imst, cax=cax, label='$\sigma_{TEC}$ [TECu]')
        else:
            posn0 = ax0.get_position()
            cax = fig.add_axes([posn0.x0+posn0.width+0.01, posn0.y0, 0.02, posn0.height])
            fig.colorbar(im1, cax=cax, label='$\sigma_{TEC}$ [TECu]')
        
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
    p.add_argument('--cmap', type = str, default = 'gray', help="colormap")
    p.add_argument('--tlim', type = str, default = None, help="time limits", nargs=2)
    p.add_argument('--lonlim', type = float, default = None, help="time limits", nargs=2)
    p.add_argument('--latlim', type = float, default = None, help="time limits", nargs=2)
    p.add_argument('--terminator', action='store_true')
    p.add_argument('--nightshade', action='store_true')
    p.add_argument('--odir', type = str, default = None, help="Path to output files")
    p.add_argument('-r', '--resolution',  type = int, default = 5, help="Time resolution, defult 5min")
    p.add_argument('--trange', type=float, default=None, help="+- minutes to plot on the map.")
    P = p.parse_args()
    
    main(date = P.date, scintfn=P.infn, clim=P.clim, tlim=P.tlim, cmap=P.cmap,
         latlim=P.latlim, lonlim=P.lonlim, terminator=P.terminator,
         nightshade=P.nightshade, resolution=P.resolution, odir=P.odir, trange=P.trange)
