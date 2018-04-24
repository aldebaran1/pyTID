#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:23:02 2018

@author: Sebastijan Mrak <smrak@bu.edu>
"""

import numpy as np
from argparse import ArgumentParser
import datetime
import yaml
import h5py
from pandas import read_hdf
from pyGnss import eclipseUtils as ec
from pyGnss import gnssUtils


def getPlainResidual(tec, Ts=1, polynom=False):
    intervals = ec.getIntervals(tec, maxgap=1, maxjump=1)
    pp = np.nan*np.ones(tec.shape[0])
    for lst in intervals:
        if lst[1]-lst[0] > 10:
            polynom_order = ec.getPolynomOrder(lst, Ts)
            pp[lst[0]:lst[1]] = ec.polynom(tec[lst[0]:lst[1]], order=polynom_order)
    polyfit = pp
    polyfit[:10] = np.nan
    polyfit[-10:] = np.nan
    
    y = tec - polyfit
    if polynom:
        return y, polyfit
    else:
        return y

def _getTEC(data, sv=[], navfile='', yamlfile='', timelim=None, 
            el_mask=30, lla=True, svbias=0, vertical=0, rxbias=0, 
            RxB=0, Ts=1):
    obstimes, tec, lla = ec.returnTEC(data, sv=sv, navfile=navfile, yamlfile=yamlfile, 
                                   timelim=timelim, el_mask=el_mask, lla=lla, 
                                   svbias=svbias, vertical=vertical, 
                                   rxbias=rxbias, RxB=RxB)
    t, tec = ec.interpolateTEC(obstimes,tec)
    t, lat = ec.interpolateTEC(obstimes,lla[0])
    t, lon = ec.interpolateTEC(obstimes,lla[1])
    return t, tec, lat, lon

if __name__ == '__main__':
    
    p = ArgumentParser()
    p.add_argument('year', type=str)
    p.add_argument('day', type=str)
    p.add_argument('cfg', type=str)
    
    P = p.parse_args()

    year = P.year
    day = P.day
    # Get processing parameters
    stream = yaml.load(open(P.cfg, 'r'))
    el_mask = stream.get('el_mask')
    Ts = stream.get('Ts')
    leap_seconds = stream.get('leap_seconds')
    latlim = stream.get('latlim')
    lonlim = stream.get('lonlim')
    processing = stream.get('processing')
    
    if processing == 'eclipse':
        overlap = stream.get('overlap') # minutes
        distance = stream.get('distance') # kilometers
    else: 
        overlap = np.nan
        distance = np.nan
    
    rxlistfn = stream.get('rxlist')
    savefn = stream.get('savefn')
    DATADIR = stream.get('datadir') + year + '/' + day + '/'
    NAVDIR = stream.get('navdir')
    if (savefn[-3:] != '.h5') or (savefn[-5:] != '.hdf5'):
        savefn += '.h5'
    
    rxlist, rxpos = ec.getRxListCoordinates(filename=rxlistfn)
    rxlist, rxpos = ec.rxFilter(rxlist, rxpos, latlim=latlim, lonlim=lonlim)
    print ("Number of receivers in the list: ", len(rxlist))
    
    sv = np.arange(1,33)
    tlim = [datetime.datetime.strptime('{} {} 0 0 0'.format(year, day),'%Y %j %H %M %S'),
            datetime.datetime.strptime('{} {} 0 0 0'.format(year, str(int(day)+1)),'%Y %j %H %M %S')]
    
    obstimes_ts = gnssUtils.datetime2posix(tlim)
    time_save = np.arange(obstimes_ts[0],obstimes_ts[1],Ts)
    residuals = np.nan*np.zeros((time_save.shape[0], sv.shape[0], len(rxlist)))
    latitudes = np.nan*np.zeros((time_save.shape[0], sv.shape[0], len(rxlist)))
    longitudes = np.nan*np.zeros((time_save.shape[0], sv.shape[0], len(rxlist)))
    
    for j in range(len(rxlist)):
        print ('Processing station: '+str(j+1)+'/'+str(len(rxlist)))
        try:
            rx = rxlist[j]
            yamlfile = DATADIR+'/'+ rx + '.yaml'
            navfile = NAVDIR + '/brdc' + day + '0.'+year[-2:]+'n'
            
            hdffile =  DATADIR + rx + '.h5'
            data = read_hdf(hdffile)
            
            for i in range(sv.shape[0]):
                try:
                    times, tec, lat, lon = _getTEC(data, sv=sv[i], navfile=navfile, yamlfile=yamlfile, 
                                           timelim=None, el_mask=el_mask, lla=True, 
                                           svbias=0, vertical=1, Ts=Ts)

                    if processing == 'normal':
                        z, polyfit = getPlainResidual(tec, Ts=Ts, polynom=True)
                    elif processing == 'eclipse':
                        totality = ec.returnTotalityPath()
                        LOS = [times, lat, lon]
                        ix, ed, errX = ec.getToatlityTouch(totality, LOS)
                        if ed.min() < distance:
                            td = [datetime.datetime.utcfromtimestamp(i) for i in times]
                            polyfit = ec.getWegihtedPolyfit(ix, np.array(td), tec, Tdelta=overlap, Ts=Ts, interval_mode=2)
                            z = tec - polyfit
                        else:
                            z, polyfit = getPlainResidual(tec, Ts=Ts, polynom=True) 
            
                    idt = np.array([np.abs(time_save - t).argmin() for t in times])
                    
                    residuals[idt,i,j] = z
                    latitudes[idt,i,j] = lat
                    longitudes[idt,i,j] = lon
                except Exception as e:
                    print (e)
        except Exception  as e:
            print (e)
    print ('Saving data......')

    h5file = h5py.File(savefn, 'w')
    h5file.create_dataset('obstimes', data=time_save)
    h5file.create_dataset('res', data=residuals)
    h5file.create_dataset('lat', data=latitudes)
    h5file.create_dataset('lon', data=longitudes)
    
    timestamp = datetime.datetime.now()
    h5file.attrs[u'processed'] = timestamp.strftime('%Y-%m-%d')
    h5file.attrs[u'number of receivers'] = len(rxlist)
    h5file.attrs[u'el_mask'] = el_mask
    h5file.attrs[u'leap_seconds'] = leap_seconds
    h5file.attrs[u'processing'] = processing
    if processing == 'eclipse':
        h5file.attrs[u'overlap'] = overlap
        h5file.attrs[u'distance'] = distance
    
    h5file.close()
