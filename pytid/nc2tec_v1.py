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
from scipy.interpolate import CubicSpline
#import matplotlib.pyplot as plt

def _mkrngs(y0, idf, gap_length=10, lim=0.05, min_length=None, max_length=None, 
            zero_mean=False, extend=0):
    gap = np.diff(np.where(idf)[0])
    i00 = np.where(idf)[0][0]
    i99 = np.where(idf)[0][-1]
    ixg = np.squeeze(np.argwhere(gap >= gap_length))
    LL = np.sort(np.hstack((ixg, ixg+1)))
    inner_limits = np.where(idf)[0][LL]
    limits = np.sort(np.hstack((i00,inner_limits,i99)))
    assert limits.size % 2 == 0
    ranges = limits.reshape(int(limits.size/2), 2)
    # Check for ranges vlidity: approx. zero mean
    if zero_mean:
        mask = []
        for i, r in enumerate(ranges):
            m_hat = np.nanmean(y0[r[0]:r[1]])
            if abs(m_hat) < lim: mask.append(i)
        if len(mask) > 0:
            mask = np.array(mask)
            ranges = ranges[mask]
    if min_length is not None:
        mask = np.squeeze(np.diff(ranges) > min_length)
        ranges = ranges[mask]
    if max_length is not None:
        mask = np.squeeze(np.diff(ranges) < max_length)
        ranges = ranges[mask]
    if len(ranges.shape) == 3:
        if isinstance(ranges, np.ndarray):
            if ranges.shape[0] != 0: 
                ranges = ranges[0]
    try:
        if extend > 0:
            start = ranges[:,0]
            ixstart = start > extend + 1
            ranges[ixstart,0] -= extend
            stop = ranges[:,1]
            ixstop = stop < (y0.size - extend - 1)
            ranges[ixstop, 1] += extend
    except:
        pass
    return ranges

def _cubicSplineFit(x, idf):
#    idf = np.isfinite(x)
    x0 = np.where(idf)[0]
    x1 = np.arange(x.size)
    CSp = CubicSpline(x0, x[idf])
    y = CSp(x1)
    return y, idf
    
def detrend(x, polynom_list=None, eps=1):
    if polynom_list is None:
        polynom_list = np.arange(0,15)
    err_list = np.nan * np.zeros(polynom_list.size)
    err_list[:3] = 9999.0
    for i in polynom_list[2:]:
        res = gu.phaseDetrend(x, order=i)
        err = np.nansum(np.abs(res)**2)
        err_list[i] = err
        D0 = abs(err_list[i-1] - err)
        D1 = abs(err_list[i-2] - err_list[i-1])
        if D0 <= eps and abs(D1 - D0) <= eps*1e-2:
            break
    return res, err_list


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('date')
    p.add_argument('rxlist', type = str, help = 'Rxlist as a .yaml file')
    p.add_argument('--elmask', type = int, default = 30)
    p.add_argument('--tlim', default = None, help = "start, stop times example 06:00 08:00", nargs=2, type=str)
    p.add_argument('-o', '--ofn', help = 'Output filename with or withou root folder.', default=None)
    p.add_argument('--ts', help = 'sampling rate', default = 30, type = int)
    p.add_argument('--cfg', help = 'Path to the config (yaml) file', default = None)
    p.add_argument('--log', help = 'If you prefer to make a .log file?', action = 'store_true')
    p.add_argument('--stec', help = 'Save slant TEC?', action = 'store_true')
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
    
    date = parser.parse(P.date)
    year = date.year
    day = date.strftime('%j')
    rxlist = os.path.expanduser(P.rxlist)
    el_mask = P.elmask
    tlim = P.tlim
    Ts = P.ts
    
    eps = 5
    weights=[1, 4, 7, 10]
    
    # Obs nav
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
    if P.log:
        logfn = os.path.splitext(savefn)[0] + '.log'
        LOG = open(logfn, 'w')
        LOG.close()
    # Correct tlim for processing purpuses:
    if P.tlim is not None:
        tlim[0] -= timedelta(hours=1)
        tlim[1] += timedelta(hours=1)
    # Output arrays
    svl = 32 #gr.load(fnc[0]).sv.values.shape[0]
    rxl = fnc.shape[0]
    
    if P.stec : slanttec = np.nan * np.zeros((tl, svl, rxl))
    residuals = np.nan * np.zeros((tl, svl, rxl))
    if Ts == 1: snr = np.nan * np.zeros((tl, svl, rxl))
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
            if P.log:
                with open(logfn, 'a') as LOG:
                    LOG.write('Processing {}/{}\n'.format(irx+1, rxl))
                    LOG.close()
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
#                    sb = satbias[sv]
                    
                    dt = D.time.values
                    tsps = np.diff(dt.astype('datetime64[s]'))[0].astype(int)
                    elv = D.el.values
                    mask = (np.nan_to_num(elv) >= el_mask)
                    
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
#                    stec += sb
#                    F = pyGnss.getMappingFunction(elv, 350)
#                    tec = stec * F
#                    tecd = gu.getPlainResidual(tec, Ts=tsps, typ='none', 
#                                               maxjump=5, weights=[1,4,7,10])
                            
                    stec_zero_bias = np.nanmin(stec)
                    stec -= stec_zero_bias + 1
                    idf_stec = np.isfinite(stec)
                    tec_ranges = _mkrngs(stec, idf_stec, gap_length=10, zero_mean=False)
                    tecd = np.nan * np.ones(stec.size)
                    for r in tec_ranges:
                        chunk = stec[r[0] : r[1]]
                        idf = np.isfinite(chunk)
                        if np.sum(np.isnan(chunk)) > 0:
                            chunk = _cubicSplineFit(chunk, idf)
                        
                        polynom_list = np.arange(0,15)
                        res, err_list = detrend(chunk, polynom_list=polynom_list, eps=eps)
                        res[~idf] = np.nan
                        if abs(np.nansum(res)) < 5:
                            tecd[r[0] : r[1]] = res
                    
                    # Print the shit
                    tecd[~mask] = np.nan
                    stec[~mask] = np.nan
    
                    ixmask = (np.nan_to_num(elv) >= el_mask)
                    idt = np.isin(t, dt[ixmask])
                    idt_reverse = np.isin(dt[ixmask], t[idt])
                    
                    if P.stec: slanttec[idt, isv, irx] = stec[ixmask][idt_reverse]
                    residuals[idt, isv, irx] = tecd[ixmask][idt_reverse]
                    if Ts == 1: snr[idt, isv, irx] = S1[ixmask][idt_reverse]
                    el[idt, isv, irx] = D.el.values[ixmask][idt_reverse]
                    az[idt, isv, irx] = D.az.values[ixmask][idt_reverse]
                except Exception as e:
                    if P.log:
                        LOG.write(str(e) + '\n')
                    else:
                        print (e)
        except Exception as e:
            if P.log:
                with open(logfn, 'a') as LOG:
                    LOG.write(str(e) + '\n')
                LOG.close()
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
    if P.log:
        with open(logfn, 'a') as LOG:
            LOG.write('Saving data...... {}\n'.format(os.path.split(savefn)[1]))
        LOG.close()
    else:
        print ('Saving data...... {}'.format(os.path.split(savefn)[1]))
    
    h5file = h5py.File(savefn, 'w')
    h5file.create_dataset('obstimes', data=th5)
    h5file.create_dataset('res', data=residuals, compression='gzip', compression_opts=9)
    if P.stec:
        h5file.create_dataset('stec', data=slanttec, compression='gzip', compression_opts=9)
    if Ts == 1: 
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
    if P.log:
        with open(logfn, 'a') as LOG:
            LOG.write('{} successfully saved.\n'.format(savefn))
        LOG.close()
    else:
        print ('{} successfully saved.'.format(savefn))
