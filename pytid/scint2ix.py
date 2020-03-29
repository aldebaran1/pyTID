#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:55:19 2019

@author: smrak
"""
import os
import yaml
import h5py
import numpy as np
from datetime import datetime
from pyGnss import pyGnss
from pyGnss import gnssUtils as gu
from pyGnss import scintillation as scint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline
from pymap3d import aer2geodetic
from argparse import ArgumentParser
import platform

if platform.system() == 'Linux':
    separator = '/'
else:
    separator = '\\'

def _runningMedian(x, N):
    n2 = int(N/2)
    iterate = np.arange(n2, x.size-n2)
    y = np.nan * np.copy(x)
    for i in iterate:
        y[i] = np.nanmedian(abs(x[i-n2:i+n2]))
    return y

def _runningMax(x,N):
    n2 = int(N/2)
    iterate = np.arange(n2, x.size-n2)
    y = np.nan * np.copy(x)
    for i in iterate:
        chunk = x[i-n2:i+n2]
        if np.sum(np.isfinite(chunk)) > 1:
            y[i] = np.nanmax(abs(chunk))
    return y

def _removeRipple(y, E = 5, L = 300, eps=False):
    std = np.nanstd(y[L:])
    envelope = _runningMax(y, N=60)
    std = np.nanmedian(envelope)
    e = E * std
    if np.where(abs(np.nan_to_num(y[:L])) >= e)[0].size > 0:
        if np.where(abs(np.nan_to_num(y[:L])) >= e)[0].size == 1:
            ex = np.where(abs(np.nan_to_num(y[:L])) >= e)[0].item() + 1
        else:
            ex = np.where(abs(np.nan_to_num(y[:L])) >= e)[0][-1] + 1
    else: 
        ex = -999
    
    if eps:
        return ex, e
    else:
        return ex

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
    if len(ranges.shape) == 3 and ranges.shape[0] != 0: ranges = ranges[0]
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

def _scintillationMask(X, X_hat, X_eps, N_median=60, min_length=60, 
                       gap_close=60*5, extend=0, 
                       diagnostic=False):
    #
    # Empty output arrays
    events = np.array([])
    Y = np.copy(X)
    # Detect the events
    # SIGMA_TEC
    # Reject suspecious data : np.nanmedian(sT) / st_hat < 1.5
    # median value of individual link has to be reasonably close to median 
    # of the receiver
    if np.nanmedian(X) / X_hat < 2:
        X_med = _runningMedian(X, N_median)
        idx = (np.nan_to_num(X_med) > np.nan_to_num(X_eps))
        idquet = np.ones(X.size, dtype = bool)
        if np.sum(idx) > 0:
            events = _mkrngs(X_med, idx, gap_length = 10, 
                             min_length = min_length, 
                             zero_mean = False, extend=extend)
            if events.size == 0:
                Y[idquet] = np.nan
                if diagnostic:
                    return Y, X_med
                else:
                    return Y
            
            if gap_close is not None:
                if len(events.shape) == 3: events = events[0]
                if events.shape[0] > 1:
                    gaps = np.empty(events.shape[0]-1, dtype=np.int32)
                    for i in np.arange(1, events.shape[0]):
                        gaps[i-1] = events[i, 0] - events[i-1, 1]
                        if events[i, 0] - events[i-1, 1] < gap_close:
                            events = np.vstack((events, [events[i-1, 1], events[i, 0]]))
            if len(events.shape) == 3: events = events[0]
            # Remove questionably low ranges. Median must be above mean
            event_mask = np.zeros(events.shape[0], dtype=bool)
            for sci, sct in enumerate(events):
                event_mask[sci] = True
            # Skip if there are no scintillation events at this place
            if events.size > 0:
                events = events[event_mask]
                for r in events:
                    idquet[r[0]:r[1]] = False
        Y[idquet] = np.nan
    if diagnostic:
        return Y, X_med
    else:
        return Y
            
def _partialProcess(dt,r, x, fs=1, fc=0.1, hpf_order=6,
                    plot_ripple = False,
                    plot_outlier = False):
    
    idf = np.isfinite(x)
    # If there are NaNs in the interval, do a cubic spline. 
    # Max gap is 10 seconds set by the "make ranges routine"
    # 1. dTEC Split
    if np.sum(np.isnan(x)) > 0:
        x0 = np.where(idf)[0]
        x1 = np.arange(x.size)
        CSp = CubicSpline(x0, x[idf])
        x_cont = CSp(x1)
    else:
        x_cont = np.copy(x)
    # 2. Tec/snr scintillation (high-pass) filtering!
    tec_hpf = gu.hpf(x_cont, fs=fs, order=hpf_order, fc=fc)
    tec_hpf[~idf] = np.nan
    
    # 3. Remove initial ripple on the scintillation time-series
    sT_exit, eps = _removeRipple(tec_hpf, E=1.5, L=300, eps = True)
    if plot_ripple:
        plt.figure()
        plt.plot(dt[r[0]:r[1]], tec_hpf, 'b')
        plt.plot([dt[r[0]], dt[r[1]]], [eps, eps], '--r')
        if sT_exit != -999: plt.plot(dt[r[0]:r[1]][:sT_exit], tec_hpf[:sT_exit], 'xr')
    if sT_exit != -999: 
        tec_hpf[:sT_exit] = np.nan
    tec_hpf_original = np.copy(tec_hpf)
    # 4. Outlier detection and removal. Still on the scintillation time-series.
    # 4.1 TEC Scintillation
    envelope = _runningMax(abs(tec_hpf), N = 10)
    median_envelope = _runningMedian(envelope, N = 120)
    outlier_margin = median_envelope + 5 * np.nanstd(tec_hpf)
    idoutlier = np.nan_to_num(abs(tec_hpf)) > outlier_margin
    
    outlier_mask = np.zeros(tec_hpf.size, dtype = bool)
    if np.nansum(idoutlier) > 0:
        outlier_intervals = _mkrngs(tec_hpf, idoutlier,
                                    max_length = 60, gap_length = 10, 
                                    zero_mean = False)
        
        if outlier_intervals.size > 0:
            if len(outlier_intervals.shape) == 3: 
                outlier_intervals = outlier_intervals[0]
            for out_ran in outlier_intervals:
                ratio = np.median(envelope[out_ran[0]:out_ran[1]+1]) / np.median(median_envelope[out_ran[0]:out_ran[1]+1])
                if np.round(ratio,1) >= 3:
                    backward = 10 if out_ran[0] > 10 else out_ran[0]
                    forward = 10 if tec_hpf.size - out_ran[1] > 10 else -1
                    outlier_mask[out_ran[0]-backward : out_ran[1]+1+forward] = True
        if plot_outlier:
            plt.figure(figsize = [8,5])
#            plt.title('2017-5-28 / Rxi: {}, svi: {}'.format(irx, isv))
            plt.plot(dt[r[0] : r[1]], tec_hpf, 'b', label = '$\delta TEC_{0.1 Hz}$')
            plt.plot(dt[r[0] : r[1]], median_envelope, 'g', label = 'env = <$\widehat{\delta TEC}>|_{10s}$')
            plt.plot(dt[r[0] : r[1]], outlier_margin, '--r', label = '$\epsilon$ = env + 4$\cdot \sigma(\delta TEC)|_{60s}$')
            plt.plot(dt[r[0] : r[1]], -outlier_margin, '--r')
            plt.plot(dt[r[0] : r[1]][outlier_mask], tec_hpf[outlier_mask], 'xr')
            plt.ylabel('$\delta$ TEC [TECu]')
            plt.xlabel('Time [UTC]')
            plt.grid(axis='both')
            plt.legend()
        tec_hpf[outlier_mask] = np.nan
    return tec_hpf, tec_hpf_original, outlier_mask

def ranges(x, idf, min_gap=10, gap_length=10, min_length=30*60, zero_mean=False):
    gap = np.diff(np.where(idf)[0])
    intervals = []
    if np.argwhere(gap >= min_gap).size > 0:
        intervals = _mkrngs(x, idf, gap_length=gap_length, 
                            min_length=min_length, zero_mean=zero_mean)
    else:
        intervals = np.array([ [np.where(idf)[0][0], 
                             np.where(idf)[0][-1]+1] ])

    if len(intervals.shape) == 3: 
        try:
            intervals = intervals[0]
        except: 
            intervals = np.array([])
    
    return intervals

def _toLLT(rxp=None, az=None, el=None, H=350):
    """
    Default height of the IPP is 350 km.
    """
    H *= 1e3
    r = H / np.sin(np.radians(el))
    lat, lon, alt = aer2geodetic(az=az, el=el, srange=r, lat0=rxp[0], lon0=rxp[1], h0=rxp[2])
    
    return lat, lon

def process(fn, odir=None, cfg=None, log=None, irxforce=None):

    ############################### Open data ##################################
    if irxforce is not None:
        irxforce = int(irxforce)
    if odir is None:
        odir = os.path.split(fn)[0] + separator
    if cfg is None:
        plot_ripple = 0
        plot_outlier = 0
        savefig = 1
        figfolder = os.path.join(odir, 'scint_plots' + separator)
        plot = 0
        
        fs = 1
        fc = 0.1
        hpf_order = 6
        H = 350
    else:
        assert os.path.splitext(cfg)[1] in ('.yml', '.yaml')
        stream = yaml.load(open(cfg, 'r'))
        plot_ripple = stream.get('plot_ripple')
        plot_outlier = stream.get('plot_outlier')
        plot = stream.get('plot')
        savefig = stream.get('savefig')
        figfolder = stream.get('figfolder')
        if figfolder is None:
            figfolder = os.path.join(odir, 'scint_plots' + separator)
        
        fs = stream.get('fs')
        fc = stream.get('fc')
        hpf_order = stream.get('hpf_order')
        H = stream.get('alt_km')
    # Output file
    if odir is None:
        odir = os.path.split(fn)[0] + separator
    ofn = odir + 'ix_' + '_'.join(os.path.split(fn)[1].split('.')[:2]) + '_{}km.h5'.format(H)
    # Dealing with duplicate file names
    if os.path.exists(ofn):
        head = os.path.splitext(ofn)[0]
        c = 0
        while os.path.exists(ofn):
            try:
                c = int(os.path.splitext(ofn)[0].split('_')[-1])
                c += 1
            except:
                c += 1
            ofn = head + '_' + str(c) + '.h5'
    if log:
        logfn = os.path.splitext(ofn)[0] + '.log'
        LOG = open(logfn, 'w')
        LOG.close()
    # Open data file
    f = h5py.File(fn, 'r')
    time = f['obstimes'][:]
    dt = np.array([datetime.utcfromtimestamp(t) for t in time])
    if irxforce is None:
        rnx = f['el'].shape[2]
    else:
        rnx = 1
    #rnx = 5
    svx = f['el'].shape[1]
    rxpall = f['rx_positions'][:]
    # New arrays
    ipp = np.nan * np.ones((dt.size, svx, rnx, 2)) # [time, SV, Rx, [lat, lon]]
    sigma_tec = np.nan * np.ones((dt.size, svx, rnx))
    snr4 = np.nan * np.ones((dt.size, svx, rnx))
    s4 = np.nan * np.ones((dt.size, svx, rnx))
    roti = np.nan * np.ones((dt.size, svx, rnx))
    if plot:
        rot = np.nan * np.ones((dt.size, svx, rnx))
#    tec_hpf = np.nan * np.ones((dt.size, svx, rnx))
    # Bookkeeping
    scint_limits = np.nan * np.zeros((rnx,2))
    receiver_std = np.nan * np.zeros((rnx,2))
    receiver_std_median = np.nan * np.zeros((rnx,2))
    
    for irx in range(rnx):
        if log:
            with open(logfn, 'a') as LOG:
                LOG.write('Processing Rx/all #{}/{}\n'.format(irx+1, rnx))
            LOG.close()
        else:
            print ('Processing Rx/all #{}/{}'.format(irx+1, rnx))
                   
        if plot:
            tec_hpf_all = np.nan * np.ones((dt.size, svx))
            snr_hpf_all = np.nan * np.ones((dt.size, svx))
            sigma_tec_all = np.nan * np.ones((dt.size, svx))
            snr4_all = np.nan * np.ones((dt.size, svx))
        # Reset to zero for each iteration
        tec_outliers = np.zeros((dt.size, svx), dtype=bool)
        snr_outliers = np.zeros((dt.size, svx), dtype=bool)
        try:
            for isv in range(svx):
                try:
                    if irxforce is not None:
                        el = f['el'][:,isv,irxforce]
                        az = f['az'][:,isv,irxforce]
                        res = f['res'][:,isv,irxforce]
                        snr = f['snr'][:,isv,irxforce]
                        rxp = rxpall[irxforce]
                    else:
                        el = f['el'][:,isv,irx]
                        az = f['az'][:,isv,irx]
                        res = f['res'][:,isv,irx]
                        snr = f['snr'][:,isv,irx]
                        rxp = rxpall[irx]
                    
                    # Compute location of the IPP
                    lat, lon = _toLLT(rxp, az=az, el=el, H=H)
                    # Get Mapping Function
                    F = pyGnss.getMappingFunction(el, h = 350)
                    # Stack into the output array
                    ipp[:, isv, irx, 0] = lat
                    ipp[:, isv, irx, 1] = lon
                except Exception as e:
                    if log:
                        with open(logfn, 'a') as LOG:
                            LOG.write('{}\n'.format(e))
                        LOG.close()
                    else:
                        print (e)
                # Check for a minum length of valid observables == 30 min
                if np.nansum(np.isfinite(res)) < 30 * 60:
                    continue
                
                tec_hpf_copy = np.nan * np.copy(res)
                rot_copy = np.nan * np.copy(res)
                roti_copy = np.nan * np.copy(res)
                sigma_tec_copy = np.nan * np.copy(res)
                snr4_copy = np.nan * np.copy(snr)
                s4_copy = np.nan * np.copy(snr)
                tec_hpf_original = np.nan * np.copy(res)
                snr_hpf_original = np.nan * np.copy(res)
                # 0.0 To ranges: Multipe visits of a satellite per day. 
                # New interval for a gap bigger than 10 samples. 
                # Minimum length of interval is 30 minutes
                # Create empty arrays
                idf_tec = np.isfinite(res)
                idf_snr = np.isfinite(snr)
                # 0.1 Do for TEC
                try:
                    tec_ranges = ranges(res, idf_tec, min_gap=10, gap_length=10, min_length=30*60, zero_mean=True)
                except:
                    tec_ranges = np.array([])
                try: 
                    snr_ranges = ranges(snr, idf_snr, min_gap=10, gap_length=10, min_length=30*60)
                except:
                    snr_ranges = np.array([])
                
                # Process TEC per intervals
                if tec_ranges.size > 0:
                    for ith_range, r in enumerate(tec_ranges):
                        # Remove to short ranges if accidentaly do occur
                        if np.diff(r) < 10: continue
                        try:
                            chunk = res[r[0] : r[1]]
                            tec_hpf, tec_hpf_original[r[0]:r[1]], tec_mask = _partialProcess(dt, r, chunk, fs=fs, fc=fc, hpf_order=hpf_order, 
                                                                                             plot_ripple=plot_ripple, plot_outlier=plot_outlier)
                            tec_outliers[r[0] : r[1], isv] = tec_mask
                            sigma_tec_copy[r[0] : r[1]] = scint.sigmaTEC(tec_hpf, N = 60)
                            tec_hpf_copy[r[0] : r[1]] = tec_hpf
                            tmp_diff = np.diff(chunk)
                            tmp_diff[tec_mask[1:]] = np.nan
                            rot_copy[r[0]+1 : r[1]] = tmp_diff
                            roti_copy[r[0]+1 : r[1]] = scint.sigmaTEC(np.diff(chunk), N=60)
                        except Exception as e:
                            if log:
                                with open(logfn, 'a') as LOG:
                                    LOG.write('{}\n'.format(e))
                                LOG.close()
                            else:
                                print (e)
                if snr_ranges.size > 0:
                    for ith_range, r in enumerate(snr_ranges):
                        # Remove to short ranges if accidentaly do occur
                        if np.diff(r) < 60: continue
                        try:
                            Schunk = snr[r[0] : r[1]].astype(np.float64)
                            snr_hpf, snr_hpf_original[r[0]:r[1]], snr_mask = _partialProcess(dt, r, Schunk, fs=fs, fc=fc, hpf_order=hpf_order,
                                                                                             plot_ripple=plot_ripple, plot_outlier=plot_outlier)
                            snr_outliers[r[0] : r[1], isv] = snr_mask
                            snr4_copy[r[0] : r[1]] = scint.sigmaTEC(snr_hpf, N = 60)
                            s4_copy[r[0] : r[1]] = scint.AmplitudeScintillationIndex(10**(Schunk/10), 60)
                        except Exception as e:
                            if log:
                                with open(logfn, 'a') as LOG:
                                    LOG.write('{}\n'.format(e))
                                LOG.close()
                            else:
                                print (e)
                # Save scintillation indices
                sigma_tec[:, isv, irx] = sigma_tec_copy
                snr4[:, isv, irx] = (snr4_copy * (F**0.9))
                s4[:, isv, irx] = (s4_copy * (F**0.9))
                roti[:, isv, irx] = roti_copy
                if plot:
                    rot[:, isv, irx] = rot_copy
                    tec_hpf_all[:,isv] = tec_hpf_original
                    snr_hpf_all[:,isv] = snr_hpf_original
                    sigma_tec_all[:,isv] = sigma_tec_copy
                    snr4_all[:,isv] = snr4_copy
            # 4. Define the scintillation event masks per receiver
            # 4.1 Define limits
            # sigma_tec: limit ------------------------------------------------------ #
            st_std = np.nanstd(sigma_tec[:, :, irx])
            st_std_tec = np.nanstd(sigma_tec[:, :, irx])
            st_hat = np.nanmedian(sigma_tec[:, :, irx])
            st_eps = 2.5 * st_hat # + st_std
            # SNR4 limit
            s4_std = np.nanstd(snr4[:, :, irx])
            s4_hat = np.nanmedian(snr4[:, :, irx])
            s4_eps = 2.5 * s4_hat # + st_std
            # 4.2 Store the limits ----------------------------------------------- #
            scint_limits[irx, 0] = st_eps
            receiver_std[irx, 0] = st_std
            receiver_std_median[irx, 0] = st_std_tec
            # ----------------------------------------------------------------------- #
            scint_limits[irx, 1] = s4_eps
            receiver_std[irx, 1] = s4_std
            receiver_std_median[irx, 1] = s4_std
            # ----------------------------------------------------------------------- #
            for isv in range(svx):
                if log:
                    with open(logfn, 'a') as LOG:
                        LOG.write('Processing scintillation sv/all {}/{}\n'.format(isv+1, svx))
                    LOG.close()
                else:
                    print ('Processing scintillation sv/all {}/{}'.format(isv+1, svx))
                sigma_tec[:,isv,irx] = _scintillationMask(sigma_tec[:,isv,irx], X_hat=st_hat, 
                                                 X_eps=st_eps, extend=0, N_median=60, 
                                                 min_length=120, gap_close=5*60)
                snr4[:,isv,irx] = _scintillationMask(snr4[:,isv,irx], X_hat=s4_hat, X_eps=s4_eps,
                                                 extend=0, min_length=120, gap_close=5*60)
                #######################################################################
                # Plot for refernce
                if plot:
                    try:
                        if np.nansum(np.isfinite(sigma_tec_all[:,isv])) > 1:
                            print ("Plotting PRN:{}".format(isv+1))
                            fig = plt.figure(figsize=[15,12])
                            ax1 = fig.add_subplot(421)
                            ax12 = ax1.twinx()
                            if irxforce is None:
                                ax1.plot(dt, f['res'][:,isv,irx], 'b', label='RXi {}; PRN {}'.format(irx, isv+1))
                                ax12.plot(dt, f['el'][:,isv,irx], 'g')
                            else:
                                ax1.plot(dt, f['res'][:,isv,irxforce], 'b', label='RXi {}; PRN {}'.format(irx, isv+1))
                                ax12.plot(dt, f['el'][:,isv,irxforce], 'g', lw=0.5)
                            ax1.set_ylabel('$\Delta$ TEC')
                            ax1.grid(axis='both')
                            ax12.set_ylabel('Elevation', color='g')
                            ax12.tick_params(axis='y', colors='green')
                            ax1.legend()
                            ax1.set_xticklabels([])
                            # Second
                            ax2 = fig.add_subplot(423, sharex=ax1)
                            ax2.plot(dt, tec_hpf_all[:,isv], 'b')
                            ax2.plot(dt[tec_outliers[:,isv]], tec_hpf_all[:,isv][tec_outliers[:,isv]], 'xr')
                            ax2.set_ylabel('$\delta TEC_{0.1 Hz}$')
                            ax2.grid(axis='both')
                            # Third
                            ax3 = fig.add_subplot(427, sharex=ax1)
                            ax3.plot(dt, sigma_tec_all[:,isv], '.b')
                            
                            i0 = np.argwhere(np.isfinite(sigma_tec_all[:,isv]))[0]
                            i1 = np.argwhere(np.isfinite(sigma_tec_all[:,isv]))[-1]
                            ax3.plot([dt[i0], dt[i1]], [st_eps, st_eps], '--r')
                            if sum(np.isfinite(sigma_tec[:,isv,irx])) > 0:
                                ax3.plot(dt, sigma_tec[:,isv,irx], '.g')
                            ax3.set_ylabel('$\sigma_{TEC}$ [TECu]')
#                            ax3.set_xlim([datetime(2017,9,8,0), datetime(2017,9,8,5)])
                            ax3.grid(axis='both')
                            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            ######################### SNR
                            ax11 = fig.add_subplot(422, sharex=ax1)
                            if irxforce is None:
                                ax11.plot(dt, f['snr'][:,isv,irx], 'b', label='RXi {}; PRN {}'.format(irx, isv+1))
                            else:
                                ax11.plot(dt, f['snr'][:,isv,irxforce], 'b', label='RXi {}; PRN {}'.format(irx, isv+1))
                            ax11.set_ylabel('SNR')
                            ax11.grid(axis='both')
                            ax11.legend()
                            # Second
                            ax21 = fig.add_subplot(424, sharex=ax1)
                            ax21.plot(dt, snr_hpf_all[:,isv], 'b')
                            ax2.plot(dt[snr_outliers[:,isv]], tec_hpf_all[:,isv][snr_outliers[:,isv]], 'xr')
                            ax21.set_ylabel('$SNR4_{0.1 Hz}$')
                            ax21.grid(axis='both')
                            # Third
                            ax31 = fig.add_subplot(426, sharex=ax1)
                            ax31.plot(dt, snr4_all[:,isv], '.b')
                            i0 = np.argwhere(np.isfinite(snr4_all[:,isv]))[0]
                            i1 = np.argwhere(np.isfinite(snr4_all[:,isv]))[-1]
                            ax31.plot([dt[i0], dt[i1]], [s4_eps, s4_eps], '--r')
                            if sum(np.isfinite(snr4[:,isv,irx])) > 0:
                                ax31.plot(dt, snr4[:,isv,irx], '.g')
                            ax31.plot(dt, s4[:,isv,irx], 'k', lw=0.5)
                            ax31.set_ylabel('SNR$_4$ [dB]')
                            ax31.grid(axis='both')
                            ax31.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            prefix = dt[0].strftime("%Y%m%d")
                            svf = '{}_rxi{}_prni{}'.format(prefix, irx,isv)
                            ax1.set_title('E($\sigma_T$) = {}'.format(st_eps))
                            ax11.set_title('E(SNR$_4$) = {}'.format(s4_eps))
                            
                            ax41 = fig.add_subplot(428, sharex=ax1)
                            ax41.plot(dt, roti[:,isv,irx], '.b')
                            ax41.set_ylabel('ROTI [TECu]')
                            ax41.grid(axis='both')
                            ax41.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            
                            ax42 = fig.add_subplot(425, sharex=ax1)
                            ax42.plot(dt, rot[:,isv,irx], 'b')
                            ax42.set_ylabel('ROT [TECu]')
                            ax42.grid(axis='both')
                            ax42.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            
                            if savefig:
                                if not os.path.exists(figfolder):
                                    import subprocess
                                    if platform.system() == 'Linux':
                                        subprocess.call('mkdir -p {}'.format(figfolder), shell=True, timeout=5)
                                    else:
                                        subprocess.call('mkdir "{}"'.format(figfolder), shell=True, timeout=5)
                                plt.savefig(figfolder+'{}.png'.format(svf), dpi=100)
                                plt.close(fig)
                        else:
                            print ("Not enoughd data from PRN:{}".format(isv+1))
                    except Exception as e:
                        print (e)
        except Exception as e:
            print (e)
        if irxforce is not None:
            break
    rxn = f['rx_name'][:]
    rxm = f['rx_model'][:]
    f.close()
    # Save to new hdf5 file
    if irxforce is None:
        if log:
            with open(logfn, 'a') as LOG:
                LOG.write('Saving data to : \n {}'.format(ofn))
            LOG.close()
        else:
            print ('Saving data to : \n {}'.format(ofn))
        
        f = h5py.File(ofn, 'w')
        gr = f.create_group('data')
        gr.create_dataset('rx_name', data = rxn, dtype='S10')
        gr.create_dataset('rx_model', data = rxm, dtype='S25')
        gr.create_dataset('time', data = time, compression = 'gzip', compression_opts = 9)
        gr.create_dataset('sigma_tec', data = sigma_tec, compression = 'gzip', compression_opts = 9)
        gr.create_dataset('snr4', data = snr4, compression = 'gzip', compression_opts = 9)
        gr.create_dataset('s4', data = s4, compression = 'gzip', compression_opts = 9)
        gr.create_dataset('roti', data = roti, compression = 'gzip', compression_opts = 9)
        gr.create_dataset('ipp', data = ipp, compression = 'gzip', compression_opts = 9)
        gr.create_dataset('rxp', data = rxpall, compression = 'gzip', compression_opts = 9)
        gr.create_dataset('scint_limits', data = scint_limits, compression = 'gzip', compression_opts = 9)
        gr.create_dataset('rxstd', data = receiver_std, compression = 'gzip', compression_opts = 9)
        gr.create_dataset('rxstdmedian', data = receiver_std_median, compression = 'gzip', compression_opts = 9)
        gr.attrs[u'altitude_km'] = H
        gr.attrs[u'hpf_fc'] = fc
        gr.attrs[u'hpf_order'] = hpf_order
        f.close()
        if log:
            with open(logfn, 'a') as LOG:
                LOG.write('Successfully saved!')
            LOG.close()
        else:
            print ('Successfully saved!')
    

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('infile')
    p.add_argument('-o', '--odir', help = 'Output directory ', default=None)
    p.add_argument('--cfg', help = 'Path to the config (yaml) file', default = None)
    p.add_argument('--log', help = 'If you prefer to make a .log file?', action = 'store_true')
    p.add_argument('--irx', help = 'Process one rx only', default=None)
    P = p.parse_args()
    
    process(fn=P.infile, odir=P.odir, cfg=P.cfg, log=P.log, irxforce=P.irx)
