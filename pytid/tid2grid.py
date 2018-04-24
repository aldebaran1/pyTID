#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:57:35 2018

@author: Sebastijan Mrak <smrak@bu.edu>
"""
import h5py
import yaml
import numpy as np
from datetime import datetime

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
    
def im2txt(im, lon=[], lat=[], res=0, fn=''):
    """
    """
    h = 'Info: lon0, lon1, lat0, lat1, res, xsize, ysize, ' + \
         str(lon[0]) +','+str(lon[1])+','+str(lat[0])+','+str(lat[1])+ \
         ','+str(res)+','+str(im.shape[0])+','+str(im.shape[1])
    np.savetxt(fn,im.flatten(), delimiter=',', header=h)
    return
        
def singleImageNew(fname,i,ylim=[0,60],xlim=[-160,0],res=0.3):
    """
    """
    f = h5py.File(fname, 'r')
    t = f['obstimes'][i]
    # Create an image grids
    xgrid, ygrid, im = makeGrid(ylim=ylim, xlim=xlim, res=res)
    
    lat = f['lat'][i][:][:]
    lon = f['lon'][i][:][:]
    res = f['res'][i][:][:]
    # Fill out the image pixels
    for sv in range(res.shape[0]):
        for j in np.where(np.isfinite(res[sv]))[0]:
            idx, idy = getImageIndex(x=lon[sv,j], y=lat[sv,j],
                                     xlim=xlim, ylim=ylim,
                                     xgrid=xgrid, ygrid=ygrid)
            # If image indexes are valid
            if np.isfinite(idx) and np.isfinite(idy):
                # Assign the value to the pixel
                if np.isnan(im[idx,idy]):
                    im[idx,idy] = res[sv,j]
                # If this is not the first value to assign, assign a
                # mean of both values
                else:
                    im[idx,idy] = (im[idx,idy] + res[sv,j]) / 2
        
    return t,xgrid,ygrid, im

def makeTheHDF(t,x,y,im,filename):
    f = h5py.File(filename, 'w')
    d = f.create_group('data')
    d.create_dataset('time',data=t)
    d.create_dataset('xgrid',data=x)
    d.create_dataset('ygrid',data=y)
    d.create_dataset('im',data=im)
    return f

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('cfg', type=str, help='path to yaml cfg file')
    P = p.parse_args()
    
    stream = yaml.load(open(P.cfg, 'r'))
    fname = stream.get('hdffn')
    savefn = stream.get('savefn')
    
    lonlim = stream.get('lonlim')
    latlim = stream.get('latlim')
    res = stream.get('resolution')
    
    ###################################################
    t = []
    im = []
    c = 0
    i = 1
    while True:
        try:
            print (i)
            time,xgrid,ygrid,image = singleImageNew(fname,i,xlim=lonlim,ylim=latlim,res=res)
            t.append(time)
            if c == 0:
                x = xgrid[:,0]
                y = ygrid[0,:]
                c+=1
            im.append(image)
            i+=1
        except:
            break
    # Save back to hdf
    f = makeTheHDF(t,x,y,im,savefn)
    timestamp = datetime.now()
    f.attrs[u'converted'] = timestamp.strftime('%Y-%m-%d')
    f.attrs[u'lonlim'] = '{} - {}'.format(lonlim[0],lonlim[1])
    f.attrs[u'latlim'] = '{} - {}'.format(latlim[0],latlim[1])
    f.attrs[u'resolution'] = res
    f.close()

#    
#ylim = [25,50]
#xlim = [-130,-65]
#res = 0.2
#
#fname = '/media/smrak/Eclipse2017/Eclipse/hdf/eclipse/eclipse232.h5'
#fname = '/media/smrak/Eclipse2017/mstid/hdf/mstid2018_013.h5'
#
#t = []
#im = []
#c = 0
#i = 1
#while True:
#    try:
#        print (i)
#        time,xgrid,ygrid,image = singleImageNew(fname,i,xlim=xlim,ylim=ylim,res=res)
#        t.append(time)
#        if c == 0:
#            x = xgrid[:,0]
#            y = ygrid[0,:]
#            c+=1
#        im.append(image)
#        i+=1
#    except:
#        break
#
#filename = '/media/smrak/Eclipse2017/mstid/hdf/mstid2018_013_processed.h5'
#f = makeTheHDF(t,x,y,im,filename)
#
#timestamp = datetime.now()
#f.attrs[u'converted'] = timestamp.strftime('%Y-%m-%d')
#f.attrs[u'lonlim'] = '{} - {}'.format(xlim[0],xlim[1])
#f.attrs[u'latlim'] = '{} - {}'.format(ylim[0],ylim[1])
#f.attrs[u'resolution'] = res
#f.close()
