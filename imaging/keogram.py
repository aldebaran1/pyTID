#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:16:01 2018

@author: Sebastijan Mrak <smrak@bu.edu>
"""
import numpy as np
import os
from glob import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
from pyGnss import gnssUtils as gu
#from sdomask import mask2d
from scipy import ndimage
import datetime

def getNeighbours(image,i,j):
    """
    Return an array of <=9 neighbour pixel of an image with a center at (i,j)
    """
    nbg = []
    for k in np.arange(i-1, i+2):
        for l in np.arange(j-1, j+2):
            try:
                nbg.append(image[k,l])
            except Exception as e:
                pass
    return np.array(nbg)

def fillPixels(im, N=1):
    """
    Fill in the dead pixels. If a dead pixel has a least 4 finite neighbour
    pixel, than replace the center pixel with a mean valuse of the neighbours
    """
    for n in range(N):
        for i in np.arange(0,im.shape[0]):
            for j in np.arange(0,im.shape[1]):
                # Check if th epixel is dead, i.e. empty
                if np.isnan(im[i,j]):
                    # Get its neighbours as a np array
                    nbg = getNeighbours(im,i,j)
                    # If there are at leas 4 neighbours, replace the value with a mean
                    if sum(np.isfinite(nbg)) >= 4:
                        ix = np.where(np.isfinite(nbg))[0]
                        avg = np.mean(nbg[ix])
                        im[i,j] = avg
    return im


def plotKeogram(t,Y,im,title='',cmap='jet',clim=[],tlim=[],ylim=[], 
                xlabel='',ylabel='',save=None,alpha=1,ptype='im',cfstep=0.01):
    
    formatter = mdates.DateFormatter('%H:%M')
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    plt.title(title)
    if ptype =='im':
        plt.pcolormesh(t,Y,im.T, cmap=cmap,alpha=alpha)
    elif ptype == 'cf':
        levels = np.arange(clim[0],clim[1]+0.01,cfstep)
        im[im<=levels[0]] = levels[0]
        im[im>=levels[-1]] = levels[-1]
        plt.contourf(t,Y,im.T, levels=levels,cmap=cmap,alpha=alpha)
    plt.clim(clim)
#    plt.colorbar()
    plt.colorbar(ticks=[clim[0], clim[0]/2, 0, clim[1]/2, clim[1]])
    if len(clim) > 0:
        plt.clim(clim)
    if len(tlim) > 0:
        plt.xlim(tlim)
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.xaxis.set(major_formatter=formatter)
    if save is not None:
        plt.savefig(save, dpi=400)
    return ax
        
def plotSpectogram(t,k,Sx,title='',cmap='viridis',clim=[],tlim=[],ylim=[], 
                   xlabel='',ylabel='',save=None, scale='lin'):
    formatter = mdates.DateFormatter('%H:%M')
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    Zm = np.ma.masked_where(np.isnan(Sx),Sx)
    if scale == 'lin':
        plt.pcolormesh(t,k,Zm.T, cmap=cmap)
    else:
        plt.pcolormesh(t,k,10*np.log10(Zm).T, cmap=cmap)
    plt.ylabel(ylabel)
    plt.clim(clim)
    plt.colorbar()
    ax.set_xticklabels(t)
    ax.xaxis.set(major_formatter=formatter)
        
def keo2txt(im, center=0, tlim=[], ylim=[], res=0, fn=''):
    """
    """
    res = round((ylim[-1]-ylim[-2]),1)
    h = 'Info: t0, t1, lim0, lim1, center, res, tsize, ysize, ' + \
         str(tlim[0]) +','+str(tlim[-1])+','+str(lim[0])+','+str(lim[-1])+ \
         ','+str(res)+','+str(im.shape[0])+','+str(im.shape[1])
    np.savetxt(fn,im.flatten(), delimiter=',', header=h)
    return

def getXYTec(filename='/media/smrak/Eclipse2017/Eclipse/hdf/eclipse/single233fill1.h5', lat=40,lon=-100):
    data = h5py.File(filename, 'r')
    time = data['data/time'].value
    xgrid = data['data/xgrid'].value
    ygrid = data['data/ygrid'].value
    im = data['data/im'][:][:][:]

    idx = abs(xgrid-lon).argmin()
    idy = abs(ygrid-lat).argmin()
    
    tec = im[:,idx,idy]
    dt = [datetime.datetime.utcfromtimestamp(t) for t in time]  
    
    return dt, tec

def interpolateTotality(lon, lat, lon0=-130,lon1=-50, order=3, resolution=0.3):
    z = np.polyfit(lon, lat, order)
    f = np.poly1d(z)
    lon_new = np.arange(lon0, lon1, resolution)
    lat_new = f(lon_new)
    return lon_new, lat_new

def getMaskKeogram(nlat=180, nlon=360,X=39,lim=[-130, -60],direction='lat',
                   EUVDIR = 'C:\\Users\\smrak\\Google Drive\\BU\\software\\sdomask\\HiResFull300\\'):
    npts = nlat*nlon
    xgrid, ygrid = mask2d.getEUVMaskCoordinates(nlat=nlat, nlon=nlon)
    
    flist = sorted(glob(EUVDIR+'*.bin'))
    
    if direction == 'lat':
        idx = np.where( (xgrid[:,0] >= lim[0])  &  (xgrid[:,0] <= lim[1]))[0]
        idy = abs(ygrid[0,:] - X).argmin()
        Y = xgrid[idx,0]
    elif direction == 'lon':
        idx = abs(xgrid[:,0] - X).argmin()
        idy = np.where( (ygrid[0,:] >= lim[0])  &  (ygrid[0,:] <= lim[1]))[0]
        Y = ygrid[0,idy]
    elif direction == 'totality':
        idx = np.where( (xgrid[:,0] >= lim[0])  &  (xgrid[:,0] <= lim[1]))[0]
        
        totality_path = h5py.File('/home/smrak/Documents/eclipse/totality.h5', 'r')
        Xt = totality_path['path/center_lon'].value
        Yt = totality_path['path/center_lat'].value - 1
        Xt, Yt = interpolateTotality(Xt,Yt,lon0=-150,lon1=-50,order=5,resolution=1)
        
        XGRID = xgrid[idx,0]
        idy = []
        for x in XGRID:
            idX = abs(x-Xt).argmin()
            Yt_approx = Yt[idX]
            tmp = abs(ygrid[0,:] - Yt_approx).argmin()
            idy.append(tmp)
        idy = np.array(idy)

    Y = xgrid[idx,0]
        
    
    
    keogram = np.nan*np.ones((len(flist), idx.shape[0]))
    keograml = np.copy(keogram)
    dt = []
    i=0
    for f in flist:
        head, tail = os.path.split(f)
        hh = int(tail[6:8])
        mm = int(tail[8:10])
        if mm == 60:
            hh = int(hh)+1
            mm = 00
        dt.append(datetime.datetime.strptime('20170821'+str(hh)+str(mm), '%Y%m%d%H%M'))
        data = np.fromfile(f,count=npts, dtype=np.float32).reshape((nlat,nlon))
        laplace = ndimage.filters.laplace(data)
        imslice = data[idy,idx] 
        lapslice = laplace[idy,idx]
        keogram[i,:] = imslice
        keograml[i,:] = lapslice
        i+=1

    return dt, Y, keogram, keograml

def getKeogram(filename, skip=1,X=40,lim=[-110,-70],direction='lat',timelim=[],
               im_filter=1, fillPixel_iter=0, integrate=2, spectogram=0):
    """
    Direction lat: Fixed LAT, look along lon
    """
    data = h5py.File(filename, 'r')
    time = data['data/time'][::skip]
    xgrid = data['data/xgrid'].value
    ygrid = data['data/ygrid'].value
    im = data['data/im'][::skip][:][:]
    #--------------------------------------------------------------------------#
    if len(timelim) > 0:
        tl_posix = gu.datetime2posix(timelim)
        idt = np.where( (time >= tl_posix[0]) & (time <= tl_posix[1]) )[0]
        im = im[idt,:,:]
        time = time[idt]
    #--------------------------------------------------------------------------#
    if direction == 'lon':
        idy = np.where( (ygrid >= lim[0]) & (ygrid<= lim[1]) )[0]
        if integrate <= 1:
            idx = abs(xgrid - X).argmin()
            keogram = im[:,idx,idy]
        elif integrate > 1:
            idx = abs(xgrid - X).argmin()
            
            idx = np.array([idx-int(integrate/2), idx, idx+int(integrate/2)])
            imstack = im[:,idx,idy[0]:idy[-1]]
            keogram = np.nan*np.zeros((imstack.shape[0], imstack.shape[2]))            
            for i in range(imstack.shape[0]):
                for j in range(imstack.shape[2]):
                    N = sum(np.isfinite(imstack[i,:,j]))
                    if N > 0:
                        keogram[i,j] = sum(np.nan_to_num(imstack[i,:,j]))/N 
        Y = ygrid[idy]
    elif direction == 'lat':
        idx = np.where( (xgrid >= lim[0]) & (xgrid<= lim[1]) )[0]
        if integrate <= 1:
            idy = abs(ygrid - X).argmin()
            keogram = im[:,idx,idy]
        elif integrate > 1:
            idy = abs(ygrid - X).argmin()
            idy = np.array([idy-int(integrate/2), idy, idy+int(integrate/2)])
            imstack = im[:,idx[0]:idx[-1],idy]
            
            keogram = np.nan*np.zeros((imstack.shape[0], imstack.shape[1]))            
            for i in range(imstack.shape[0]):
                for j in range(imstack.shape[1]):
                    N = sum(np.isfinite(imstack[i,j,:]))
                    if N > 0:
                        keogram[i,j] = sum(np.nan_to_num(imstack[i,j,:]))/N 
        Y = xgrid[idx]
    elif direction == 'totality':
        totality_path = h5py.File('/home/smrak/Documents/eclipse/totality.h5', 'r')
        Xt = totality_path['path/center_lon'].value
        Yt = totality_path['path/center_lat'].value - 1
        Xt, Yt = interpolateTotality(Xt,Yt,lon0=-150,lon1=-50,order=5,resolution=0.3)
        
        idx = np.where( (xgrid >= lim[0]) & (xgrid<= lim[1]) )[0]
        idy = []

        keogram = np.nan*np.ones((time.shape[0],xgrid[idx].shape[0]))
        for x in xgrid[idx]:
            idX = abs(x-Xt).argmin()
            Yt_approx = Yt[idX]
            tmp = abs(ygrid-Yt_approx).argmin()
            idy.append(tmp)
        idy = np.array(idy)
        
        if integrate <= 1:
            keogram = im[:,idx,idy]
        else:
            imkeogram = np.nan*np.ones((time.shape[0],xgrid[idx].shape[0], integrate+1))
            intrange = np.arange(-int(integrate/2), int(integrate/2)+1)
            for i in range(intrange.shape[0]):
                imkeogram[:,:,i] = im[:,idx,idy+intrange[i]]
            for i in range(imkeogram.shape[0]):
                for j in range(imkeogram.shape[1]):
                    N = sum(np.isfinite(imkeogram[i,j,:]))
                    if N > 0:
                        keogram[i,j] = sum(np.nan_to_num(imkeogram[i,j,:]))/N 
        Y = xgrid[idx]
        
    dt = [datetime.datetime.utcfromtimestamp(t) for t in time]  
    if fillPixel_iter > 0:
        keogram = fillPixels(keogram,N=fillPixel_iter)
    if im_filter:
        keogram = ndimage.median_filter(keogram, 3)
    
    
    if spectogram:
        
        if direction == 'lat':
            n = Y.shape[0]
            d = round(ygrid[1] - ygrid[0],1)*111*np.cos(np.deg2rad(X))
        else:
            n = Y.shape[0]
            d = round(ygrid[1] - ygrid[0],1)*111
        
        k = np.fft.fftfreq(n,d=d)
        k = 2*np.pi*k[1:int((n-1)/2)]
        specto = np.nan*np.zeros((keogram.shape[0], k.shape[0]))
#        print (keogram.shape, Y.shape, specto.shape, n, k.shape)
        for j in range(keogram.shape[0]):
            x = np.nan_to_num(keogram[j])
            Sx = np.abs(np.fft.fft(x))
            Sx = Sx[1:int((n-1)/2)]
            specto[j,:] = Sx
    
        return [dt, keogram, Y], [k,specto]
    else:
        return [dt, keogram, Y]

def plotMaskKeogram(t,Y,z,laplace=None,cmap='gray',alpha=0.9,lw=1):
    z[z==1] = np.nan
    z = np.ma.masked_where(np.isnan(z),z)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    formatter = mdates.DateFormatter('%H:%M')
    plt.pcolormesh(t,Y,z.T, alpha=alpha, cmap='gray')
    plt.colorbar()
    if laplace is not None:
#        levels = np.linspace(-0.03,0.03,10)
        levels = np.linspace(0.005,0.04,4)
        laplace[laplace>=levels[-1]] = levels[-1]
        plt.contour(t,Y,laplace.T, levels, colors='r',colorwidths=lw)
    
    ax.xaxis.set(major_formatter=formatter)

day = 234
filename = '/media/smrak/Eclipse2017/Eclipse/hdf/eclipse/single'+str(day)+'_02.h5'
filename = 'E:\\single234_02.h5'
#savefolder = '/media/smrak/Eclipse2017/Eclipse/keogram/new/'
savefolder = '/home/smrak/Documents/eclipse/GRL3/'
savefolder = 'E:\\surav\\'
#euvdir = '/home/smrak/Documents/eclipse/HiResFull300/'
 

#direction = 'totality'
#X = np.arange(30,51,2)
direction = 'lat'
X = [37.7]
lim = [-120, -70]
#direction = 'lon'
#X =[-90,-85,-95]
#lim = [25,50]

#t_mask, Y_mask, keogram_mask, keograml_mask = getMaskKeogram(X=X[0],lim=lim,direction=direction,EUVDIR=euvdir)
#plotMaskKeogram(t_mask,Y_mask,keogram_mask,laplace=abs(keograml_mask),cmap='Greens')
#keofn = savefolder + str(day) + direction + str(X[0])
#plt.savefig(keofn+'penumbra2.png', dpi=400)
fillpixel_iter = 1
skip=1
integrate=5
im_filter = True

clim = [-0.15,0.15]
timelim = [datetime.datetime(2017,8,22,0,0,0), datetime.datetime(2017,8,22,12,0,0)]
#timelim=[]

for X in X:
    #dt, tec = getXYTec(filename,lat,lon)
    [dt, keogram, Y], [k,Sx] = getKeogram(filename,X=X,lim=lim,direction=direction,skip=skip,
                                fillPixel_iter=fillpixel_iter,im_filter=im_filter,
                                timelim=timelim, integrate=integrate,
                                spectogram=True)
    
    im = np.ma.masked_where(np.isnan(keogram),keogram)
    
    if direction == 'lat':
        ylabel = 'lon [deg]'
    else:
        ylabel = 'lat [deg]'
    keofn = savefolder + str(day) + direction + str(X)
#    spectname = savefolder + 'specto' + str(day) + direction + str(X)
    ax = plotKeogram(dt,Y[:im.shape[1]],im,clim=clim, tlim=[], title='Fixed '+direction+'-- Center: '+str(X),
                cmap='jet',alpha=1,ptype='cf',cfstep=0.01)#, save=keofn+'grl3.png') #xlabel='time [UT]',ylabel=ylabel, 

#    cmap1 = colors.LinearSegmentedColormap.from_list("", ['black', 'magenta'])
#    levels = np.linspace(0.005,0.04,3)
#    iml = abs(keograml_mask)
#    iml[iml>=levels[-1]] = levels[-2]
#    ax.contour(t_mask,Y_mask,iml.T, levels, colors='k',linewidths=3,alpha=0.9)
    plt.savefig(keofn+'30s.png', dpi=300)
#    plotSpectogram(dt,k,Sx, clim=[0,2],xlabel='time [UT]',ylabel='',scale='lin')
#    plt.savefig(spectname+'g3.png', dpi=200)
#    keo2txt(keogram, center=X,tlim=timelim,ylim=[Y[0],Y[-1]],res=Y[-1]-Y[-2],fn=keofn+'txt')