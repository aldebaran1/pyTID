#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:27:43 2019

@author: smrak
"""
import os
import yaml
import h5py
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

def makeGrid(ylim=[25,50],xlim=[-110,-80],res=0.5):
    """
    Make a grid for an image with a given boundaries and resolution
    """
    xd = abs(xlim[0] - xlim[1]) / res * 1j
    yd = abs(ylim[0] - ylim[1]) / res * 1j
    xgrid, ygrid = np.mgrid[xlim[0]:xlim[1]:xd, ylim[0]:ylim[1]:yd]
    z = np.nan*np.zeros((xgrid.shape[0], xgrid.shape[1]))
    
    return xgrid, ygrid, z

def getImageIndex(x, y, xlim, ylim, xgrid, ygrid):
    """
    find and return a pixel location on the image to map the LOS value. find the
    pixel which minimizes the distance in x and y direction
    """
    if x > xlim[0] and x < xlim[1] and y > ylim[0] and y < ylim[1]:
        idy = abs(ygrid[0,:] - y).argmin()
        idx = abs(xgrid[:,0] - x).argmin()
    else:
        idy = np.nan
        idx = np.nan
    return idx, idy

def makeImage(dtec, longitude, latitude, im):
    for isv in range(dtec.shape[0]):
            for irx in np.where(np.isfinite(dtec[isv]))[0]:
                idx, idy = getImageIndex(x=longitude[isv,irx], y=latitude[isv,irx],
                                         xlim=lonlim, ylim=latlim,
                                         xgrid=xgrid, ygrid=ygrid)
                # If image indexes are valid
                if np.isfinite(idx) and np.isfinite(idy):
                    # Assign the value to the pixel
                    if np.isnan(im[idx,idy]):
                        im[idx,idy] = dtec[isv,irx]
                    # If this is not the first value to assign, assign a
                    # mean of both values
                    else:
                        im[idx,idy] = (im[idx,idy] + dtec[isv,irx]) / 2
    return im

def makeTheHDF(t,x,y,im,filename):
    f = h5py.File(filename, 'w')
    d = f.create_group('data')
    d.create_dataset('time',data=t)
    d.create_dataset('xgrid',data=x)
    d.create_dataset('ygrid',data=y)
    d.create_dataset('im',data=im, compression='gzip', compression_opts=9)
    return f

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('fname', type=str)
    p.add_argument('--ofn', type=str, default=None)
    p.add_argument('--cfg', type=str, default=None)
    p.add_argument('--mode', type=str, help='Input coordinates: lla, or aer?', default='lla')
    p.add_argument('--altkm', type=float, help='Projection altitude in km', default=350)
    p.add_argument('-r', '--resolution', type=float, help='Maps resolution, default is from cfg file', default=None)
    p.add_argument('-x', type=float, help='longitude limits. Default from cfg file', default=None, nargs=2)
    p.add_argument('-y', type=float, help='latitude limits. Default from cfg file', default=None, nargs=2)
    P = p.parse_args()
    
    fname = P.fname   
    
    savefn = P.ofn
    mode = P.mode
    

    if P.cfg is not None:
        cfg = P.cfg 
        stream = yaml.load(open(cfg, 'r'))
        lonlim =  stream.get('lonlim')
        latlim = stream.get('latlim')
        resolution = stream.get('resolution')
    else:
        resolution = P.resolution
        lonlim = P.x
        latlim = P.y
    # Create an image grids
    xgrid, ygrid, im = makeGrid(ylim=latlim, xlim=lonlim, res=resolution)
    ###################################################
    images = []
    ###################################################
    f = h5py.File(fname, 'r')
    time = f['obstimes'][:]
    res = f['res'][:]
    if mode == 'aer':
        from pymap3d import aer2geodetic
        r1 = (P.altkm*1e3) / np.sin(np.radians(f['el'][:]))
            
        ipp_lla = aer2geodetic(az=f['az'][:], el=f['el'][:], srange=r1, 
                               lat0=f['rx_positions'][:,0], 
                               lon0=f['rx_positions'][:,1], 
                               h0=f['rx_positions'][:, 2])
        lat = ipp_lla[0]
        lon = ipp_lla[1]
        
    else:
        lat = f['lat'][:]
        lon = f['lon'][:]
    
    for i in range(time.shape[0]):
        print ("{}/{}".format(i+1, time.shape[0]))
        try:
            imtemp = makeImage(dtec=res[i], latitude = lat[i], longitude = lon[i], im=np.nan*im)
            images.append(imtemp)
        except Exception as e:
            print (e)
    
    if savefn is None:
        folder = os.path.split(fname)[0]
        root = os.path.split(fname)[1].rstrip('.h5')
        rr = str(resolution).replace('.', '')
        filename = 'grid/grid_{}_altkm_{}_res_{}.h5'.format(root, int(P.altkm), rr)
        savefn = folder + filename
    elif not savefn.endswith('.h5'):
        root = os.path.split(fname)[1].rstrip('.h5')
        rr = str(resolution).replace('.', '')
        addon = '{}_altkm_{}_res_{}.h5'.format(root, int(P.altkm), rr)
        savefn += addon
    f = makeTheHDF(time,xgrid[:,0],ygrid[0,:],images,savefn)
    timestamp = datetime.now()
    f.attrs[u'converted'] = timestamp.strftime('%Y-%m-%d')
    f.attrs[u'lonlim'] = '{} - {}'.format(lonlim[0],lonlim[1])
    f.attrs[u'latlim'] = '{} - {}'.format(latlim[0],latlim[1])
    f.attrs[u'resolution'] = resolution
    f.attrs[u'altkm'] = P.altkm
    f.close()
