#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:27:21 2017

@author: Sebastijan Mrak <smrak@bu.edu>
"""
import numpy as np
from pyGnss import eclipseUtils as ec
import datetime
from pandas import read_hdf
import matplotlib.pyplot as plt
import os
import h5py
import multiprocessing
import yaml
from argparse import ArgumentParser

DATADIR = '/media/smrak/Eclipse2017/Eclipse/cors/'
NAVDIR = '/media/smrak/Eclipse2017/Eclipse/nav/'
SAVEDIR = '/media/smrak/Eclipse2017/Eclipse/hdf/'

def plotLOS(t,y, td=False, title='', save=False):
    if not td:
        t = [datetime.datetime.utcfromtimestamp(i) for i in t]
    fig = plt.figure(figsize=(12,8))
    if title is not '':
        plt.title(title)
    plt.plot(t,y,'.b')
    plt.show()
    plt.close(fig)
    
def plotTecRes(Tt, TEC, Tr, RES, polynom=None, td=False,  title='', save=False):
    """
    
    """
    if not td:
        try:
            tt = [datetime.datetime.utcfromtimestamp(i) for i in Tt]
        except:
            pass
        try:
            tr = [datetime.datetime.utcfromtimestamp(i) for i in Tt]
        except:
            pass
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    ax1.plot(tt, TEC, '.b', lw=3)
    if polynom is not None:
        ax1.plot(tt, polynom, 'r', lw=1)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(tr, RES, 'r')
    ax2.plot([tr[0], tr[-1]], [0,0], 'k')
    if title is not '':
        ax1.set_title(title)
    if save:
        plt.savefig('ref_plot/{}.png'.format(title), dpi=800)
    else:
        plt.show()
        plt.close(fig)
# ---------------------------------------------------------------------------- #
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
            RxB=0, Ts=1, interpolate=False):
    t, tec, lla = ec.returnTEC(data, sv=sv, navfile=navfile, yamlfile=yamlfile, 
                               timelim=timelim, el_mask=el_mask, lla=lla, 
                               svbias=svbias, vertical=vertical, 
                               rxbias=rxbias, RxB=RxB)
    t_corr, tec = ec.correctSampling(t, tec, fs=Ts)
    t_corr, lat = ec.correctSampling(t, lla[0], fs=Ts)
    t_corr, lon = ec.correctSampling(t, lla[1], fs=Ts)
    
    if interpolate:
        tec = ec.interpolateTEC(t_corr, tec)
    
    return t_corr, tec, lat, lon

# ---------------------------------------------------------------------------- #
def individualSite(rx='', RxBias=False, polynom_order=None):
    """
    Sebastijan Mrak
    The function takes a set of global variables, computes the lat-lon-residuals
    for each SV in list and returns 4 NP arrays of time, lat, lon, res for all
    lines of sight. The funstion is designed to be used with multi-threading or
    multi-processing loops. The only in-function argument is receiver name as str.
    """
    
    global sTEC, timelim, day, year, RES, los_clobber, clobber_mode
    global sv, vTEC, plot, save, interpolate, el_mask, decimate, Ts
    
    if decimate is not None:
        hdffile =  DATADIR +'/' + rx + day + '0_' + str(decimate) + '.h5'
        Ts = decimate
    else:
        hdffile =  DATADIR + '/' + rx + day + '0.h5'

    yamlfile = DATADIR+'/'+ rx + day + '0.yaml'
    navfile = NAVDIR + '/brdc' + day + '0.'+year[-2:]+'n'
    
    data = read_hdf(hdffile)
    
    if sv == 'all':
        sv = list(range(1,33))
        sv.pop(4)
    time = []
    residuals = []
    latitude = []
    longitude = []
    for i in range(len(sv)):
        try:
            # Test case 1
            if sTEC:
                t, tec, lat, lon = _getTEC(data, sv=sv[i], navfile=navfile, yamlfile=yamlfile, 
                                           timelim=None, el_mask=el_mask, lla=True, 
                                           svbias=0, vertical=0, Ts=Ts, interpolate=interpolate)
                # Plot TEC only
                if not RES and plot:
                    plotLOS(t, tec, title=rx+'- SV: '+str(sv[i])+ '. sTEC', save=save)
                # Get resiudals
                else:
                    # Get residuals in one piece
                    if not los_clobber:
                        z, polyfit = getPlainResidual(tec, Ts=Ts, polynom=True)
                        if plot:
                            plotTecRes(t, tec, t, z, polynom=polyfit,
                                       title=rx+'- SV: '+str(sv[i])+ '. sTEC-clobberOFF', save=save)
                    # Clobber the line
                    else:
                        totality = ec.returnTotalityPath()
                        LOS = [t, lat, lon]
                        ix, ed, errX = ec.getToatlityTouch(totality, LOS)
                        if ed.min() < 1000:
                            td = [datetime.datetime.utcfromtimestamp(i) for i in t]
                            polyfit = ec.getWegihtedPolyfit(ix, np.array(td), tec, Tdelta=20, Ts=Ts, interval_mode=clobber_mode)
                            z = tec - polyfit
                        else:
                            z, polyfit = getPlainResidual(tec, Ts=Ts, polynom=True)
                        if plot:
                            plotTecRes(t, tec, t, z, polynom=polyfit,
                                       title=rx+'- SV: '+str(sv[i])+ '. sTEC-clobberON', save=save)
                # Save the result
                time.append(t)
                residuals.append(z)
                latitude.append(lat)
                longitude.append(lon)
            # Test case 2
            if vTEC:
                t, tec, lat, lon = _getTEC(data, sv=sv[i], navfile=navfile, yamlfile=yamlfile, 
                                           timelim=None, el_mask=el_mask, lla=True, 
                                           svbias=1, vertical=1, Ts=Ts, interpolate=interpolate)
                # Plot TEC only
                if not RES and plot:
                    plotLOS(t, tec, title=rx+'- SV: '+str(sv[i])+ '. vTEC-no RxB', save=save)
                # Get resiudals
                else:
                    # Get residuals in one piece
                    if not los_clobber:
                        z = getPlainResidual(tec, Ts=Ts)
                        if plot:
                            plotTecRes(t, tec, t, z, 
                                       title=rx+'- SV: '+str(sv[i])+ '. vTEC no RxB-clobberOFF', save=save)
                    # Clobber the line
                    else:
                        totality = ec.returnTotalityPath()
                        LOS = [t, lat, lon]
                        ix, ed, errX = ec.getToatlityTouch(totality, LOS)
                        if ed.min() < 1000:
                            td = [datetime.datetime.utcfromtimestamp(i) for i in t]
                            polyfit = ec.getWegihtedPolyfit(ix, np.array(td), tec, Tdelta=20, Ts=Ts, interval_mode=clobber_mode)
                            z = tec - polyfit
                        else:
                            z = getPlainResidual(tec, Ts=Ts)
                        if plot:
                            plotTecRes(t, tec, t, z, 
                                       title=rx+'- SV: '+str(sv[i])+ '. vTEC no RxB-clobberON', save=save)
                # Save the result
                time.append(t)
                residuals.append(z)
                latitude.append(lat)
                longitude.append(lon)
            # Test case 3
            if RxBias:
                if np.isin(sv[i], [2,6,12,19]):
                    RxB = [datetime.datetime(2017,8,21,17,0,0), 10]
                elif np.isin(sv[i], [5]):
                    RxB = [datetime.datetime(2017,8,21,19,0,0), 10]
                t, tec, lat, lon = _getTEC(data, sv=sv[i], navfile=navfile, yamlfile=yamlfile, 
                                           timelim=None, el_mask=el_mask, lla=True, 
                                           svbias=0, vertical=1, rxbias=1, RxB=RxB,
                                           Ts=Ts, interpolate=interpolate)
                # Plot TEC only
                if not RES and plot:
                    plotLOS(t, tec, title=rx+'- SV: '+str(sv[i])+ '. vTEC', save=save)
                # Get resiudals
                else:
                    # Get residuals in one piece
                    if not los_clobber:
                        z, polyfit = getPlainResidual(tec, Ts=Ts, polynom=True)
                        if plot:
                            plotTecRes(t, tec, t, z, polynom=polyfit,
                                       title=rx+'- SV: '+str(sv[i])+ '. vTEC-clobberOFF', save=save)
                    # Clobber the line
                    else:
                        totality = ec.returnTotalityPath()
                        LOS = [t, lat, lon]
                        ix, ed, errX = ec.getToatlityTouch(totality, LOS)
                        if ed.min() < 1000:
                            td = [datetime.datetime.utcfromtimestamp(i) for i in t]
                            polyfit = ec.getWegihtedPolyfit(ix, np.array(td), tec, Tdelta=20, Ts=Ts, interval_mode=clobber_mode)
                            z = tec - polyfit
                            if plot:
                                plotTecRes(t, tec, t, z, polynom=polyfit,
                                           title=rx+'- SV: '+str(sv[i])+ '. vTEC-clobberON', save=save)
                        else:
                            z = getPlainResidual(tec, Ts=Ts)
                            if plot:
                                plotTecRes(t, tec, t, z, 
                                           title=rx+'- SV: '+str(sv[i])+ '. vTEC-clobberON', save=save)
                # Save the result
                time.append(t)
                residuals.append(z)
                latitude.append(lat)
                longitude.append(lon)
        except Exception as e:
            print (e)
            
    return np.array(time), np.array(residuals), np.array(latitude), np.array(longitude)

def makeEmptyArrays(a,b):
    """
    Create empty NP arrays
    """
    x = np.nan*np.zeros((a, b))
    y = np.nan*np.zeros((a, b))
    z = np.nan*np.zeros((a, b))
    return x,y,z
            
def save2HDF(time=[],residuals=[],latitude=[],longitude=[],timelist=[],
             rx='', minute=[0,0], overwrite=False):
    """
    Save processed data for a single receiver into a set of HDF files, splitted into
    time intervals specified in the configuration file. 
    Location of the saved files is specified in global varaibles SAVEDIR and the 
    file prefix is given by the configuration file.
    timelist must be in the form of [ [], [], [] ] i.e. list(list()))
    """
    global day, year, fnprefix
    #Save data file by file given the time interavls 
    for tl in timelist:
        fname= SAVEDIR+fnprefix+str(day)+'_'+str(tl[0])+'.h5'
        # 24:00 is invalid, replace it with 23:59
        if tl[1] == 24:
            time_array = ec.createTimeArray([datetime.datetime.strptime('{} {} {} {}'.format(
                        str(year), str(day), str(tl[0]), str(minute[0])),'%Y %j %H %M'),
                                        datetime.datetime.strptime('{} {} {} {}'.format(
                        str(year), str(day), '23', '59'),'%Y %j %H %M')])
            
        else:
            time_array = ec.createTimeArray([datetime.datetime.strptime('{} {} {} {}'.format(
                        str(year), str(day), str(tl[0]), str(minute[0])),'%Y %j %H %M'),
                                        datetime.datetime.strptime('{} {} {} {}'.format(
                        str(year), str(day), str(tl[1]), str(minute[1])),'%Y %j %H %M')])
        # Create initial file and write in obstimes time stamps
        if overwrite:
            h5file = h5py.File(fname, 'w')
            h5file.create_dataset('obstimes', data=time_array)
            h5file.close()
        else:
            if not os.path.exists(fname):
                h5file = h5py.File(fname, 'w')
                h5file.create_dataset('obstimes', data=time_array)
                h5file.close()
        # Define empty NaN value arrays
        lat,lon,res = makeEmptyArrays(np.shape(time_array)[0], np.shape(time)[0])
        # Assign values into Nan arrays satellite by satellite
        for i in range(time.shape[0]):
            idt_array = np.where(np.isin(time_array, time[i]))[0]
            t_in = time_array[idt_array]
            idt_obs = np.where(np.isin(time[i], t_in))[0]
            lat[idt_array,i] = latitude[i][idt_obs]
            lon[idt_array,i] = longitude[i][idt_obs]
            res[idt_array,i] = residuals[i][idt_obs]
        # Write data to file for a given receiver
        try:
            h5file = h5py.File(fname, 'a')
            gr = h5file.create_group(rx)
            gr.create_dataset('lat', data=lat)
            gr.create_dataset('lon', data=lon)
            gr.create_dataset('res', data=res)
            h5file.close()
        except Exception as e:
            print (e)
        
def parallelHandler(rx):
    global timelist
    t, r, lat, lon = individualSite(rx=rx)
    save2HDF(t, r, lat, lon,timelist=timelist, rx=rx)
    
if __name__ == '__main__':
    
    global day, sTEC, vTEC, los_clobber, clobber_mode, timelist
    global sv, year, interpolate, el_mask, decimate, Ts, fnprefix
    
    p = ArgumentParser()
    p.add_argument('year', type=str)
    p.add_argument('day', type=str)
    p.add_argument("-m", '--mode', default='test', type=str)
    p.add_argument("--cfg", "--cfg", type=str)
    
    P = p.parse_args()
    
    try:
        year = P.year
        day = P.day
        DATADIR +=year+'/'+day+'/'
        stream = yaml.load(open(P.cfg, 'r'))
        rxlist = stream.get('rxlist')
        latlim = stream.get('latlim')
        lonlim = stream.get('lonlim')
        rxstart = stream.get('rxstart')
        if rxlist == 'all':
            rx, rxpos = ec.getRxListCoordinates()
            rxlist, rxl_positions = ec.rxFilter(rx, rxpos, latlim=latlim, lonlim=lonlim)
            if rxstart is not None:
                ix = np.where(rxlist == rxstart)[0]
#                print (rxlist.shape[0])
#                print (rxstart)
#                print (ix[0])
                rxlist = rxlist[ix[0]:]
        print ('First receiver: ', rxlist[0])
        print ('All stations: ', rxlist.shape[0])
            
        interpolate = stream.get('tec_interpolate')
        los_clobber = stream.get('los_clobber')
        clobber_mode = stream.get('clobber_mode')
        plot = stream.get('plot')
        save = stream.get('save')
        sv = stream.get('sv')
        timelist = stream.get('timelist')
        sv = stream.get('sv')
        sTEC = stream.get('sTEC')
        vTEC = stream.get('vTEC')
        RES  = stream.get('RES')
        los_clobber = stream.get('los_clobber')
        clobber_mode = stream.get('clobber_mode')
        savefolder = stream.get('save_folder')
        fnprefix = stream.get('fn_prefix')
        SAVEDIR += savefolder
        
        el_mask = stream.get('elevation_mask')
        decimate = stream.get('decimate')
        Ts = stream.get('Ts')
        
        for rx in rxlist:
            print (rx)
            if P.mode == 'test':
                t, r, lat, lon = individualSite(rx=rx)
                if save:
                    save2HDF(t, r, lat, lon,timelist=timelist, rx=rx)
            elif P.mode == 'parallel':
                p = multiprocessing.Process(target=parallelHandler, args=(rx,))
                p.start()
                p.join(50)
            else:
                raise ('Enter -m argument, either "test" or "parallel"')
    except Exception as e:
        raise(e)