#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:57:09 2019

@author: smrak
"""
from pyGnss import pyGnss
from pyGnss import gnssUtils as gu
from datetime import datetime, timedelta
import georinex as gr
import numpy as np
from glob import glob
from dateutil import parser
import yaml
import os
import h5py
from argparse import ArgumentParser
#import matplotlib.pyplot as plt

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('date')
    p.add_argument('rxlist', type = str, help = 'Rxlist as a .yaml file')
    p.add_argument('--elmask', type = int, default = 30)
    p.add_argument('--tlim', default = None, help = "start, stop times example 06:00 08:00", nargs=2, type=str)
#    p.add_argument('-i', '--altkm', type = int, help = 'Pierce points altitude in kilometers.', default=350)
    p.add_argument('-o', '--ofn', help = 'Output filename with or withou root folder.', default=None)
    p.add_argument('--ts', help = 'sampling rate', default = 30, type = int)
    p.add_argument('--cfg', help = 'Path to the config (yaml) file', default = None)
    p.add_argument('--log', help = 'If you prefer to make a .log file?', action = 'store_true')
    P = p.parse_args()
    
    # GLOBAL VARIABLES
    if P.cfg is None:
        OBSFOLDER = '/media/smrak/gnss/obs/'
        NAVFOLDER = '/media/smrak/gnss/nav/'
        SBFOLDER = '/media/smrak/gnss/jplg/'
        SAVEFOLDER = '/media/smrak/gnss/hdf/'
    else:
        yamlcfg = yaml.load(open(P.cfg, 'r'))
        OBSFOLDER = yamlcfg.get('obsfolder')
        NAVFOLDER = yamlcfg.get('navfolder')
        SBFOLDER = yamlcfg.get('sbfolder')
        SAVEFOLDER = yamlcfg.get('savefolder')
    
    flog = open()
    date = parser.parse(P.date)
    year = date.year
    day = date.strftime('%j')
    rxlist = os.path.expanduser(P.rxlist)
#    ipp_alt = float(P.altkm)
    el_mask = P.elmask
    tlim = P.tlim
    Ts = P.ts
    
    weights=[1, 4, 7, 10]
    
    # Obs nav
#    if Ts == 1: OBSFOLDER += 'highrate/'
    nc_root = os.path.join(OBSFOLDER, str(year))
    # Filter input files
    stream = yaml.load(open(rxlist, 'r'))
    rxn = stream.get('rx')
    rx_total = stream.get('total')
    nc_folder = os.path.join(nc_root, str(day)) + '/'
    nc_list = np.array(sorted(glob(nc_folder + '*.nc')))
    nc_rx_name = np.array([os.path.split(r)[1][:4] for r in nc_list])
    idn = np.isin(nc_rx_name, rxn)
    fnc = nc_list[idn]
    # Nav file
    nav_root = NAVFOLDER
    fnav = os.path.join(nav_root, 'brdc' + str(day) + '0.' + str(year)[2:] + 'n')
    # jplg file
    jplg_root = SBFOLDER
    fjplg = os.path.join(jplg_root, 'jplg' + str(day) + '0.' + str(year)[2:] + 'i')
    satbias = pyGnss.getSatBias(fjplg)
    # Processing options
    satpos = True
#    ipp = True
    args = ['L1', 'L2']
    #Common time array
    if tlim is None:
        t0 = datetime.strptime('{} {}'.format(year,int(day)),'%Y %j')
        t1 = datetime.strptime('{} {}'.format(year,int(day) + 1),'%Y %j')
    else:
        assert len(tlim) == 2
        t0 = datetime.strptime('{} {}-{}'.format(year,int(day),tlim[0]),'%Y %j-%H:%M')
        t1 = datetime.strptime('{} {}-{}'.format(year,int(day),tlim[1]),'%Y %j-%H:%M')
    t = np.arange(t0, t1, Ts, dtype='datetime64[s]') #datetime64[s]
    tlim = [t0, t1]
    tl = t.shape[0]
    
    # Savename
    if P.ofn is None:
        sfn = str(year) + '_' + tlim[0].strftime('%m%dT%H%M') + '-' + tlim[1].strftime('%m%dT%H%M') + '_' + os.path.split(rxlist)[1] + '_' + str(el_mask) +'el_' + str(Ts) + 's.h5' 
        savefn = os.path.join(SAVEFOLDER, sfn)
    else:
        if os.path.isfile(P.ofn): 
            assert os.file.splitext(P.ofn)[1] in ('.h5', '.hdf5')
            savefn = P.ofn
        elif os.path.isdir(P.ofn):
            assert os.file.splitext(P.ofn)[1] in ('.h5', '.hdf5')
            savefn = os.path.join(P.ofn, os.path.split(rxlist)[1] + '_' + str(year) + '.h5')
        else:
            assert os.file.splitext(P.ofn)[1] in ('.h5', '.hdf5')
            savefn = os.path.join(SAVEFOLDER, P.ofn)
    # Open log file is choosen so
    if P.log is not None:
        logfn = os.path.splitext(savefn)[0] + '.log'
        LOG = open(logfn, 'w')
    # Correct tlim for processing purpuses:
    if P.tlim is not None:
        tlim[0] -= timedelta(hours=1)
        tlim[1] += timedelta(hours=1)
    # Output arrays
    svl = 32 #gr.load(fnc[0]).sv.values.shape[0]
    rxl = fnc.shape[0]
    
    if Ts > 10 : slanttec = np.nan * np.zeros((tl, svl, rxl))
    residuals = np.nan * np.zeros((tl, svl, rxl))
    snr = np.nan * np.zeros((tl, svl, rxl))
    el = np.nan * np.zeros((tl, svl, rxl))
    az = np.nan * np.zeros((tl, svl, rxl))
    rxpos = np.nan * np.zeros((rxl, 3))
    for irx, fnc in enumerate(fnc):
        # New Array
        TEC = np.nan * np.zeros(t.shape[0], dtype=np.float16)
        TECD = np.nan * np.zeros(t.shape[0], dtype=np.float16)
        try:
            svlist = gr.load(fnc).sv.values
            navdata = gr.load(fnav)
            navdatatime = navdata.time.values
            rxpos[irx] = gr.load(fnc).position_geodetic
            rxn[irx] = nc_rx_name[irx]
            if P.log is not None:
                LOG.write('{}/{}\n'.format(irx+1, rxl))
            else:
                print ('{}/{}'.format(irx+1, rxl))
            for isv, sv in enumerate(svlist):
                if isv > 32: 
                    continue
                if not 'G' in sv:
                    continue
                
                try:
                    D = pyGnss.dataFromNC(fnc,fnav,sv=sv,tlim=tlim,el_mask=el_mask-10, satpos=True)#, ipp=True, ipp_alt = ipp_alt)
                    idel = D['idel'].values
                    sb = satbias[sv]
                    
                    dt = D.time.values
                    tsps = np.diff(dt.astype('datetime64[s]'))[0].astype(int)
                    elv = D.el.values
                    
                    if np.isfinite(D['C1'].values[idel]).shape[0] < (15 / (60/tsps)): 
                        # If shorter than 15 minutes, skip
                        continue
                    
                    C1 = D['C1'].values
                    C1[~idel] = np.nan
                    C2 = D['P2'].values
                    C2[~idel] = np.nan
                    L1 = D['L1'].values
                    L1[~idel] = np.nan
                    L2 = D['L2'].values
                    L2[~idel] = np.nan
                    try:
                        S1 = D['S1'].values
                        S1[~idel] = np.nan
                    except:
                        S1 = np.nan * np.copy(C1)
                    
                    # Intervals
                    stec = np.nan * np.ones(dt.shape[0])
                    ixin, intervals = pyGnss.getIntervals(L1,L2,C1,C2, maxgap=1)
                    for r in intervals:
                        if r[-1] - r[0] < 50:
                            continue
                        else:
                            stec[r[0]:r[-1]] = pyGnss.slantTEC(C1[r[0]:r[-1]], C2[r[0]:r[-1]], 
                                                          L1[r[0]:r[-1]], L2[r[0]:r[-1]])
                    stec += sb
                    F = pyGnss.getMappingFunction(elv, 350)
                    tec = stec * F
                    tecd = gu.getPlainResidual(tec, Ts=tsps, typ='none', 
                                               maxjump=5, weights=[1,4,7,10])
    
                    ixmask = (np.nan_to_num(elv) >= el_mask)
                    idt = np.isin(t, dt[ixmask])
                    idt_reverse = np.isin(dt[ixmask], t[idt])
                    
                    if Ts > 10: slanttec[idt, isv, irx] = stec[ixmask][idt_reverse]
                    residuals[idt, isv, irx] = tecd[ixmask][idt_reverse]
                    snr[idt, isv, irx] = S1[ixmask][idt_reverse]
                    el[idt, isv, irx] = D.el.values[ixmask][idt_reverse]
                    az[idt, isv, irx] = D.az.values[ixmask][idt_reverse]
                except Exception as e:
                    if P.log is not None:
                        LOG.write(e + '\n')
                    else:
                        print (e)
        except Exception as e:
            if P.log is not None:
                LOG.write(e + '\n')
            else:
                print (e)
    th5 = gu.datetime2posix(t.astype(datetime))
    
    # Dealing with duplicate file names
    if os.path.exists(savefn):
        head = os.path.splitext(savefn)[0]
        c = 0
        while os.path.exists(savefn):
            try:
                c = int(os.path.splitext(savefn)[0].split('_')[-1])
                c += 1
            except:
                c += 1
            savefn = head + '_' + str(c) + '.h5'
            
    # putting the output file togather
    if P.log is not None:
        LOG.write('Saving data...... {}\n'.format(os.path.split(savefn)[1]))
    else:
        print ('Saving data...... {}'.format(os.path.split(savefn)[1]))
    
    h5file = h5py.File(savefn, 'w')
    h5file.create_dataset('obstimes', data=th5)
    h5file.create_dataset('res', data=residuals, compression='gzip', compression_opts=9)
    if Ts > 10:
        h5file.create_dataset('stec', data=slanttec, compression='gzip', compression_opts=9)
    h5file.create_dataset('snr', data=snr, compression='gzip', compression_opts=9)
    h5file.create_dataset('az', data=az, compression='gzip', compression_opts=9)
    h5file.create_dataset('el', data=el, compression='gzip', compression_opts=9)
    h5file.create_dataset('rx_positions', data=rxpos, compression='gzip', compression_opts=9)
    
    timestamp = datetime.now()
    h5file.attrs[u'processed'] = timestamp.strftime('%Y-%m-%d')
    h5file.attrs[u'number of receivers'] = rxl
    h5file.attrs[u'el_mask'] = el_mask
    h5file.attrs[u'weights'] = weights
    
    h5file.close()
    if P.log is not None:
        LOG.write('{} successfully saved.\n'.format(savefn))
    else:
        print ('{} successfully saved.'.format(savefn))
