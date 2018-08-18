#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:45:36 2018

@author: smrak
"""

import h5py
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
from pyGnss import gnssUtils

def interpolateTEC(t,y,order=2):
    x = np.arange(0,t.shape[0])
    mask = np.isfinite(y)
    z = np.polyfit(x[mask], y[mask], order)
    f = np.poly1d(z)
    y_fit = f(x)
    y[~mask] = y_fit[~mask]
    return t, y

def ht(x, fs=1/30):
    analytic_signal = signal.hilbert(x)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)
    
    return amplitude_envelope, instantaneous_frequency

def getdTEC(data,ilat,ilon,N=3,interpolate=True,lpf=True,fc=0.0015,T=30):
    xgrid = data['data/xgrid'].value
    ygrid = data['data/ygrid'].value
    print (ilon,ilat)
    posixtime = data['data/time'].value
    dt = np.array([datetime.utcfromtimestamp(t) for t in posixtime])
    if isinstance(ilat,list) and isinstance(ilon,list) and (len(ilat) == len(ilon)):
        for j in range(len(ilat)):
            idx = abs(xgrid - ilon[j]).argmin()
            idy = abs(ygrid - ilat[j]).argmin()
            dTECarr = data['data/im'][:,idx-N+1:idx+N,idy-N+1:idy+N]
            dTEC1 = np.array([np.nanmedian(dTECarr[i]) for i in range(dTECarr.shape[0])])
            if interpolate:
                t0,dTEC1 = interpolateTEC(dt, dTEC1, order=2)
            if lpf:
                dTEC1 = gnssUtils.lpf(dTEC1,fs=1/T, fc=fc)
            if j == 0:
                dTEC = dTEC1
            else:
                dTEC = np.vstack((dTEC,dTEC1))
    else:
        idx = abs(xgrid - ilon).argmin()
        idy = abs(ygrid - ilat).argmin()
        print (idx,idy)
        dTECarr = data['data/im'][:,idx-N+1:idx+N,idy-N+1:idy+N]
        dTEC = np.array([np.nanmedian(dTECarr[i]) for i in range(dTECarr.shape[0])])
        if interpolate:
            t0,dTEC = interpolateTEC(dt, dTEC, order=2)
        if lpf:
            dTEC = gnssUtils.lpf(dTEC,fs=1/T, fc=fc)
    return dt, dTEC

def tdft(t,y,T=30,nfft=1024,Nw=240,Nskip=1,window='hamming'):
    Wn = signal.get_window(window,Nw)
    f = np.fft.fftfreq(nfft,d=T) * 1e3 # to mHz
    f = f[1:int(nfft/2)]
    
    Treducted = t[:-Nw]
    Tspecto = Treducted[::Nskip] + timedelta(seconds=Nw/2*T)
    
    for i in np.arange(0,y.shape[0]-Nw,Nskip):
        Stmp = np.fft.fft(y[i:i+Nw]*Wn,n=nfft)
        Sx1 = abs(Stmp[1:int(nfft/2)])**2
        if i == 0:
            Sx = Sx1
        else:
            Sx = np.vstack((Sx,Sx1))
            
    return Tspecto, f, Sx

def plotDTEC(t,y,xlim=[],ylim=[],figsize=(8,5),color=[],save='',title=''):
    formatter = mdates.DateFormatter('%H:%M')
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    ax = fig.add_subplot(111)
    if y.ndim == 1:
        plt.plot(t, y,'b')
    else:
        for i in range(y.shape[0]):
            plt.plot(t, y[i],c=color[i])
    if len(xlim) == 2:
        plt.xlim(xlim)
    if len(ylim) == 2:
        plt.tlim(ylim)
    ax.xaxis.set(major_formatter=formatter)
    if save != '':
        plt.savefig(save, dpi = 400)
    else:
        plt.show()
    
def plotSpectrogram(x,y,z,ylim=[],xlim=[],clim=[-10,2],scale='log',
                    figsize=(12,8),save=''):
    formatter = mdates.DateFormatter('%H:%M')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if scale == 'log':
        z = np.log(z)
    plt.pcolormesh(x, y, np.log(z).T, cmap='jet')
    if len(xlim) == 2:
        plt.xlim(xlim)
    if len(ylim) == 2:
        plt.ylim(ylim)
    if len(ylim) == 2:
        plt.clim(clim)
    plt.colorbar()
    ax.xaxis.set(major_formatter=formatter)
    if save != '':
        plt.savefig(save, dpi = 400)
    else:
        plt.show()
        
def feed2hdf(t,tec,lat=[],lon=[],fn='/home/smrak/Documents/eclipse/surav/233.h5'):
    if isinstance(t[0], datetime):
        t = gnssUtils.datetime2posix(t)
    f = h5py.File(fn,'w')
    gr = f.create_group('data')
    gr.create_dataset('times',data=t)
    gr.create_dataset('dtec', data=tec)
    gr.attrs['lon'] = lon
    gr.attrs['lat'] = lat
    f.close()
    
    return

folder = '/media/smrak/Eclipse2017/Eclipse/hdf/eclipse/'
datafn = 'single233_03.h5'
#datafn = 'single234_02.h5'
fn = folder + datafn

ilon = -89.2
ilat = 37.7
timelim = [datetime(2017,8,21,4,0,0),
           datetime(2017,8,21,23,59,0)]
dTEClim = [datetime(2017,8,21,0,0,0),
           datetime(2017,8,22,0,0,0)]
spectrolim = [datetime(2017,8,21,6,0,0),
           datetime(2017,8,21,21,0,0)]

data = h5py.File(fn, 'r')
dt, dTEC = getdTEC(data,ilon=ilon,ilat=ilat,N=3,fc=0.002)
plotDTEC(dt,dTEC,xlim=dTEClim, title='89.2W, 37.7N')
feed2hdf(dt,dTEC,lat=ilat,lon=ilon)
#dt0, dTEClist = getdTEC(data,ilon=[-95,-90,-85],ilat=[42,40,38],N=3,fc=0.002)

#idt = np.where( (dt>=timelim[0]) & (dt<=timelim[1]) )[0]
#times = dt[idt]
#dTEC = dTEC[idt]
#Tspecto, f, Sx = tdft(times,dTEC,nfft=2048)


#plotDTEC(times,dTEC,xlim=dTEClim,color=['b','r','k'])#,
#save='/home/smrak/Documents/eclipse/GRL3/dtec.png')
#plotSpectrogram(Tspecto,f,Sx,ylim=[0,5],xlim=spectrolim,clim=[-3,5],figsize=(10,5))#,
#                save='/home/smrak/Documents/eclipse/GRL3/periodogram.png')
#
#t0,y = interpolateTEC(times, dTEC,order=2)
#yf = gnssUtils.lpf(y,fs=1/30, fc=0.0015)
#y = yf



#hamp, hfreq = ht(y)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.plot(times[1:],abs(hfreq), 'b')
#ax.xaxis.set(major_formatter=formatter)
#
#T = 30
#N = 360
#Wn = signal.get_window('hamming',N)
#Nskip = 1
#nfft = 2048
#fs = 1/T
#
#f = np.fft.fftfreq(nfft,d=30) * 1e3
##f = np.fft.fftshift(f)
#f = f[1:int(nfft/2)] 
#Tpf = 1/f/60
#
#Treducted = times[:-N]
#Tspecto = Treducted[::Nskip] + timedelta(seconds=N/2*T)
#
#
#for i in np.arange(0,y.shape[0]-N,Nskip):
#    Stmp = np.fft.fft(y[i:i+N]*Wn,n=nfft)
#    Sx1 = abs(Stmp[1:int(nfft/2)])**2
##    Sx1 = abs(Stmp)**2
#    if i == 0:
#        Sx = Sx1
#    else:
#        Sx = np.vstack((Sx,Sx1))

#a = np.flip(Sx,1)
#formatter = mdates.DateFormatter('%H:%M')
#fig = plt.figure(figsize=(12,8))
#ax = fig.add_subplot(111)
#plt.pcolormesh(Tspecto, f, np.log(Sx).T, cmap='jet')
##lvl = np.linspace(0,30,30)
##plt.contourf(Tspecto,f,Sx.T,levels=lvl, cmap='jet')
#plt.ylim([0.15,5])
##plt.ylim([0,0.008])
#plt.clim([-10,2])
#plt.colorbar()
#ax.xaxis.set(major_formatter=formatter)

#for i in range(y.shape[0] - N):
    

#fs = 1/30
#F, T, Sxx = signal.spectrogram(y, fs, nfft=512,window=signal.get_window('hamming', 240), nperseg=240, noverlap=239)
#plt.pcolormesh(T, F, Sxx)
