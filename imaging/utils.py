# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:36:29 2019

@author: smrak@bu.edu
"""
import numpy as np

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
#        skip = int(np.floor((3+n)/2))
        skip = 1
        starti = 0
        startj = 0
        forwardi = int(np.floor(0.6*X))
        backwardi = int(np.floor(0.4*X))
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
                            avg = np.nanmean(nbg[ix])
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
                            avg = np.nanmean(nbg[ix])
                            im[i,j] = avg
        else:
            for j in np.arange(startj, Y, skip):
                for i in np.arange(starti, forwardi, skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy, i, j, N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix = np.where(np.isfinite(nbg))[0]
                            avg = np.nanmean(nbg[ix])
                            im[i,j] = avg

            for j in np.arange(Y, 0, -skip):
                for i in np.arange(X, backwardi, -skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix =np. where(np.isfinite(nbg))[0]
                            avg = np.nanmean(nbg[ix])
                            im[i,j] = avg
    return im

def makeGrid(ylim=[25,50],xlim=[-110,-80],res=0.5):
    """
    Make a grid for an image with a given boundaries and resolution
    """
    xd = abs(xlim[0] - xlim[1]) / res * 1j
    yd = abs(ylim[0] - ylim[1]) / res * 1j
    xgrid, ygrid = np.mgrid[xlim[0]:xlim[1]:xd, ylim[0]:ylim[1]:yd]
    z = np.nan*np.zeros((xgrid.shape[0], xgrid.shape[1]))
    
    return xgrid, ygrid, z

def getImageIndex(x, y, xgrid, ygrid):
    """
    find and return a pixel location on the image to map the LOS value. find the
    pixel which minimizes the distance in x and y direction
    """
    xlim = [xgrid.min(), xgrid.max()]
    ylim = [ygrid.min(), ygrid.max()]
    if x > xlim[0] and x < xlim[1] and y > ylim[0] and y < ylim[1]:
        idy = abs(ygrid[0,:] - y).argmin()
        idx = abs(xgrid[:,0] - x).argmin()
    else:
        idy = np.nan
        idx = np.nan
    return idx, idy

def makeImage(dtec, longitude, latitude, im, xgrid, ygrid):
    for isv in range(dtec.shape[0]):
            for irx in np.where(np.isfinite(dtec[isv]))[0]:
                idx, idy = getImageIndex(x=longitude[isv,irx], y=latitude[isv,irx],
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