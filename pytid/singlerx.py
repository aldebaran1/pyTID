#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:37:32 2018

@author: Sebastijan Mrak <smrak@bu.edu>
"""

from datetime import datetime
from pandas import read_hdf
from numpy import diff, array
from pyGnss import eclipseUtils as ec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def getGPSdata(rxname='cro1',sv=5,el_mask=30,RxB=[datetime(2017,8,21,18,0,0), 25],
              svbias=True,rxbias = True,vertical = True,lla = True,day=233,
              timelim = [datetime(2017,8,21,17,0,0), datetime(2017,8,21,22,0,0)],
              residualmode='normal',Tdelta=10,distance=2000,
              obsfolder = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\Eclipse2017\\GRL2\\gps\\',
              navfolder = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\Eclipse2017\\GRL2\\gps\\'):
    """
    """
    
    yamlfile = obsfolder+rxname+str(day)+'0.'+'yaml'
    hdffile = obsfolder+rxname+str(day)+'0.'+'h5'
    navfile = navfolder+'brdc'+str(day)+'0.'+'17n'
    svbiasfile = navfolder+'jplg'+str(day)+'0.'+'yaml'

    lla = True
    vertical = True
    rxbias = True
    svbias=True

    data = read_hdf(hdffile)
    obstimes, tec, lla = ec.returnTEC(data, sv=sv, navfile=navfile, yamlfile=yamlfile, 
                                   timelim=timelim, el_mask=el_mask, lla=lla, 
                                   svbias=svbias, vertical=vertical, 
                                   rxbias=rxbias, RxB=RxB,svbiasfile=svbiasfile)
    tts = diff(obstimes)/1e9
    Ts = int(tts[0])
    print (int(tts[0]))
    t, tec1 = ec.interpolateTEC(obstimes,tec,Ts=Ts)
    if residualmode == 'normal':
        z, polyfit = ec.getPlainResidual(tec1, Ts=30, polynom=True)
    elif residualmode == 'eclipse':
        totality = ec.returnTotalityPath()
        LOS = [t, lla[0], lla[1]]
        ix, ed, errX = ec.getToatlityTouch(totality, LOS)
        if ed.min() < distance:
            td = [datetime.utcfromtimestamp(i) for i in t]
            print (len(td), tec.shape)
            polyfit = ec.getWegihtedPolyfit(ix, array(td), tec1, Tdelta=Tdelta, Ts=Ts, interval_mode=2)
            z = tec1 - polyfit
        else:
            z, polyfit = ec.getPlainResidual(tec, Ts=Ts, polynom=True)
    dt = [datetime.utcfromtimestamp(i) for i in t]
    return dt, tec1, z, polyfit, lla

day = 233
obsfolder = '/media/smrak/Eclipse2017/Eclipse/2017/'+str(day)+'/'
navfolder = '/media/smrak/Eclipse2017/Eclipse/nav/'
savefolder = '/home/smrak/Documents/eclipse/GRLMrak/rev3/'
rxname = 'ialn'
el_mask = 20
sv = 2
Tdelta = 10
RxB = [datetime(2017,8,21,16,0,0), 6]
timelim = [datetime(2017,8,21,15,0,0), datetime(2017,8,21,22,0,0)]
plotlim = [datetime(2017,8,21,16,30,0), datetime(2017,8,21,21,30,0)]
#residualmode = 'normal'

t, tec, res_n, polyfit_n, lla = getGPSdata(rxname=rxname,obsfolder=obsfolder,navfolder=navfolder,
                                       el_mask=el_mask,sv=sv,day=day,RxB=RxB,timelim=timelim,
                                       Tdelta=Tdelta,residualmode='normal')

t, tec, res_e, polyfit_e, lla = getGPSdata(rxname=rxname,obsfolder=obsfolder,navfolder=navfolder,
                                       el_mask=el_mask,sv=sv,day=day,RxB=RxB,timelim=timelim,
                                       Tdelta=Tdelta,residualmode='eclipse')



formatter = mdates.DateFormatter('%H:%M')
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
plt.plot(t, tec, 'b', lw=6)
plt.plot(t, polyfit_n, 'g', lw=4)
plt.plot(t, polyfit_e, 'r', lw=3)
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = fig.add_subplot(212, sharex=ax1)
plt.plot(t, res_n, 'b')
plt.plot(t, res_e, 'r')
plt.plot([t[0],t[-1]], [0,0], '--k', lw=0.5)

#ax1.set_xlim(plotlim)
ax1.tick_params(direction='in')
ax2.tick_params(direction='in')
plt.subplots_adjust(hspace = .1)
ax1.xaxis.set(major_formatter=formatter)

plt.savefig(savefolder+rxname+'1.png', png=400)