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
import subprocess
from glob import glob
import xarray as xr
from dateutil import parser
import yaml
import os
import h5py
from argparse import ArgumentParser
import warnings

warnings.filterwarnings('ignore')

svmap = {'G01': 0, 'G02': 1, 'G03': 2, 'G04': 3, 'G05': 4,
         'G06': 5, 'G07': 6, 'G08': 7, 'G09': 8, 'G10': 9,
         'G11': 10, 'G12': 11, 'G13': 12, 'G14': 13, 'G15': 14,
         'G16': 15, 'G17': 16, 'G18': 17, 'G19': 18, 'G20': 19,
         'G21': 20, 'G22': 21, 'G23': 22, 'G24': 23, 'G25': 24,
         'G26': 25, 'G27': 26, 'G28': 27, 'G29': 38, 'G30': 29,
         'G31': 30, 'G32': 31,
         'E01': 40, 'E02': 41, 'E03': 42, 'E04': 43, 'E05': 44,
         'E06': 45, 'E07': 46, 'E08': 47, 'E09': 48, 'E10': 49,
         'E11': 50, 'E12': 51, 'E13': 52, 'E14': 53, 'E15': 54,
         'E16': 55, 'E17': 56, 'E18': 57, 'E19': 58, 'E20': 59,
         'E21': 60, 'E22': 61, 'E23': 62, 'E24': 63, 'E25': 64,
         'E26': 65, 'E27': 66, 'E28': 67, 'E29': 68, 'E30': 69,
         'E31': 70, 'E32': 71,}
svall = np.array(['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08',
                 'G09', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16',
                 'G17', 'G18', 'G19', 'G20', 'G21', 'G22', 'G23', 'G24',
                 'G25', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31', 'G32'])

def get_nav_files(navfolder, times):
    if not isinstance(times.dtype, datetime):
        times = times.astype('datetime64[s]').astype(datetime)
    days = np.unique([date.date() for date in times]).astype('datetime64[s]').astype(datetime)

    for i, day in enumerate(days):
        day = day
        year = day.year
        doy = day.strftime('%j')
        yy = day.strftime('%y')
     
        gps_ts = (day - datetime(1980, 1, 6)).total_seconds()
        wwww = int(gps_ts / 60 /60 / 24 / 7)
        weekday = (day.weekday() + 1 ) % 7
        wwwwd = str(wwww) + str(weekday)
        
        tmp = glob(navfolder + os.sep + f"GFZ*{year}{doy}*SP3")[0] if len(glob(navfolder + f"GFZ*{year}{doy}*SP3")) > 0 else None
        if tmp is None:
            tmp = glob(navfolder + os.sep + f"IGS*{year}{doy}*SP3")[0] if len(glob(navfolder + f"IGS*{year}{doy}*SP3")) > 0 else None
        if tmp is None:
            tmp = glob(navfolder + os.sep + f"igs*{doy}*.{yy}sp3")[0] if len(glob(navfolder + f"igs*{doy}*.{yy}sp3")) > 0 else None
        if tmp is None:
            tmp = glob(navfolder + os.sep + f"gfz*{wwwwd}.sp3")[0] if len(glob(navfolder + f"gfz*{wwwwd}.sp3")) > 0 else None
        if tmp is None:
            tmp = glob(navfolder + os.sep + f"brdc*{doy}*{yy}n")[0] if len(glob(navfolder + f"brdc*{doy}*{yy}n")) > 0 else None    
        if tmp is None:
            print (f"Navigation file wasn't found. Sepcified nav_file {navfolder}")
        
        if i == 0:
            fsp3 = tmp
        else:
            fsp3 = np.hstack((fsp3, tmp))
    return fsp3

def do_one(fnc, i, f, window_size1, window_size2, window_size3):
    global leap_seconds, fsp3, polynom_list, E0, el_mask, t, ts, compute_times, Hipp, use
    L = len(fnc)
    try:
        for l in range(1,L):
            if l == 1:
                D = xr.concat((gr.load(fnc[0], use = use, fast=0, interval=ts), gr.load(fnc[l], use = use, fast=0, interval=ts)), dim='time')
            else:
                D = xr.concat((D, gr.load(fnc[l], use = use, fast=0, interval=ts)), dim='time')
        if leap_seconds is None:
            try:
                leap_seconds = D.leap_seconds
            except:
                pass
        D = D.sel(time=((D.time.values >= np.datetime64(compute_times[0])) & (D.time.values <= np.datetime64(compute_times[-1]))))
        svlist = D.sv.values
        dt = np.array([np.datetime64(ttt) for ttt in D.time.values]).astype('datetime64[s]').astype(datetime) #- timedelta(seconds=leap_seconds)
        tsps = D.interval
        maxjump = 1.6 + (np.sqrt(tsps) - 1)
        N1 = int((60/tsps)*window_size1)
        N2 = int((60/tsps)*window_size2)
        N3 = int((60/tsps)*window_size3)
        NROTI = int((60/tsps) * 5) # ROTI over 5 min
        eps = E0 * np.sqrt(30/tsps)
        rxp = np.array(D.position_geodetic)
        
        # TODO Correct GPST too UTC time when calling AER in pyGnss
        AER = pyGnss.getAER(dt, rxp, fsp3, svlist=svlist, H=Hipp)
        idel = AER[:,:,1] < el_mask
        AER[idel] = np.nan
        STEC = pyGnss.getSTEC(fnc=D, fsp3=None, el_mask=el_mask,
                              maxgap=1, maxjump=maxjump, return_aer=0)
        STEC[idel] = np.nan 
        ROTI = pyGnss.getROTI(STEC, ts=tsps, N = NROTI)
        DCB = pyGnss.getDCBfromSTEC(STEC, AER, el_mask=el_mask)
        STECcorr = STEC - DCB
        
        if os.path.split(fsp3[0])[1][:4] == 'brdc':
            DTEC = pyGnss.getDTEC2(STECcorr, eps=eps, tsps=tsps, polynom_list=polynom_list)
            DTECsg1 = pyGnss.getDTECsg_from_VTEC(STECcorr, N=N1, order=1)
            DTECsg2 = pyGnss.getDTECsg_from_VTEC(STECcorr, N=N2, order=1)
            DTECsg3 = pyGnss.getDTECsg_from_VTEC(STECcorr, N=N3, order=1)
        else:
            F = pyGnss.getMappingFunction(AER[:,:,1], 350)
            VTEC = STECcorr * F
            DTEC = pyGnss.getDTEC2(VTEC, eps=eps, tsps=tsps, polynom_list=polynom_list)
            DTECsg1 = pyGnss.getDTECsg_from_VTEC(VTEC, N=N1, order=1)
            DTECsg2 = pyGnss.getDTECsg_from_VTEC(VTEC, N=N2, order=1)
            DTECsg3 = pyGnss.getDTECsg_from_VTEC(VTEC, N=N3, order=1)
        SNR = pyGnss.getCNR(D, fsp3=None, el_mask=el_mask, H=350)
        SNR[idel] = np.nan
        try:
            rxmodel = D.rxmodel
        except:
            rxmodel = 'none'
        # Resampling
        idt_reverse = np.isin(dt, t)
        isv_reverse = np.isin(svall, svlist)
        # Missed samples?
        idt_original = np.isin(t, dt[idt_reverse])
        
        if dt[idt_reverse].size != t.size:    
            for j, isv in enumerate(np.where(isv_reverse)[0]):
                with h5py.File(f,'r+') as ds:
                    ds['stec'][idt_original, isv, i] = STECcorr[idt_reverse, j]
                    ds['roti'][idt_original, isv, i] = ROTI[idt_reverse, j]
                    ds['snr'][idt_original, isv, i] = SNR[idt_reverse, j]
                    ds['res'][idt_original, isv, i] = DTEC[idt_reverse, j]
                    ds['res_sg1'][idt_original, isv, i] = DTECsg1[idt_reverse, j]
                    ds['res_sg2'][idt_original, isv, i] = DTECsg2[idt_reverse, j]
                    ds['res_sg3'][idt_original, isv, i] = DTECsg3[idt_reverse, j]
                    ds['az'][idt_original, isv, i] = AER[:, j, 0][idt_reverse]
                    ds['el'][idt_original, isv, i] = AER[:, j, 1][idt_reverse]
                ds.close()
        else:
            with h5py.File(f,'r+') as ds:
                ds['stec'][:, isv_reverse, i] = STECcorr[idt_reverse]
                ds['roti'][:, isv_reverse, i] = ROTI[idt_reverse]
                ds['snr'][:, isv_reverse, i] = SNR[idt_reverse]
                ds['res'][:, isv_reverse, i] = DTEC[idt_reverse]
                ds['res_sg1'][:, isv_reverse, i] = DTECsg1[idt_reverse]
                ds['res_sg2'][:, isv_reverse, i] = DTECsg2[idt_reverse]
                ds['res_sg3'][:, isv_reverse, i] = DTECsg3[idt_reverse]
                ds['az'][:, isv_reverse, i] = AER[:, :, 0][idt_reverse]
                ds['el'][:, isv_reverse, i] = AER[:, :, 1][idt_reverse]
            ds.close()
        del STEC, AER, ROTI, DCB, F, VTEC, STECcorr, DTEC, DTECsg1, DTECsg2, DTECsg3,SNR
        return D.position_geodetic, D.filename[:4], rxmodel
        
    except Exception as e:
        # print (f"Error in {fnc}; {e}")
        return str(e)

def main_gps(date, obsfolder, navfolder, rxlist, tlim, odir, window_size1, window_size2, window_size3, log):
    global leap_seconds, fsp3, t, ts, use, compute_times
    assert os.path.exists(obsfolder)
    assert os.path.exists(navfolder)
    assert os.path.exists(rxlist)
    if isinstance(date, str):
        date = parser.parse(date)
        
    year = date.year
    yeara = (date - timedelta(days=1)).year
    yearz = (date + timedelta(days=1)).year
    doy = date.strftime('%j')
    mmdd = date.strftime('%m%d')
    mmdda = (date - timedelta(days=1)).strftime('%m%d')
    mmddz = (date + timedelta(days=1)).strftime('%m%d')
    
    # # Filter input files
    # stream = yaml.safe_load(open(rxlist, 'r'))
    # rxn = np.array(stream.get('rx'))
    # # Obs files => Path to
    # assert os.path.exists(obsfolder), "Folder with observation files does not exists."
    # nc_list = np.array(sorted(glob(obsfolder + '*crx') + glob(obsfolder + '*rnx') + glob(obsfolder + '*.*d') + glob(obsfolder + '*.*o')))
    # nc_rx_name, iux = np.unique(np.array([os.path.split(r)[1][:4].lower() for r in nc_list]), return_index=1)
    # idn = np.isin(nc_rx_name, rxn)
    # fn_list = nc_list[iux][idn]
    # # Nav file
    # fsp3 = os.path.join(navfolder, 'igs' + str(doy) + '0.' + str(year)[2:] + 'sp3')
    # # Break at the beginning 
    # assert os.path.exists(fsp3), "Cant find the sp3 file"
    
    #Common time array
    if tlim is None:
        t0 = date #datetime.strptime('{} {}'.format(year,int(doy)),'%Y %j')
        t1 = date + timedelta(days=1) # datetime.strptime('{} {}'.format(year,int(doy) + 1),'%Y %j')
        compute_times = np.arange(t0 - timedelta(hours=3), t1 + timedelta(hours=3.001), ts, dtype='datetime64[s]')
        t = np.arange(t0 - timedelta(minutes=10), t1 + timedelta(minutes=10), ts, dtype='datetime64[s]') #datetime64[s]
    else:
        assert len(tlim) == 2
        t0 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[0]),'%Y %j-%H:%M')
        t1 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[1]),'%Y %j-%H:%M')
        compute_times = np.arange(t0, t1, ts, dtype='datetime64[s]')
        t = np.arange(t0, t1, ts, dtype='datetime64[s]') #datetime64[s]
    tlim = [t0, t1]
    tl = t.size
    
    # Filter input files
    stream = yaml.safe_load(open(rxlist, 'r'))
    rxn = np.array(stream.get('rx'))
    # Obs files => Path to
    assert os.path.exists(obsfolder), "Folder with observation files does not exists."
    # list of input files 
    nc_list = np.array(sorted(glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' +  '*crx') + glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' + '*rnx') + glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' + '*.*d') + glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' + '*.*o')))
    nc_lista = np.array(sorted(glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' +  '*crx') + glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*rnx') + glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*.*d') + glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*.*o')))
    nc_listz = np.array(sorted(glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' +  '*crx') + glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*rnx') + glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*.*d') + glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*.*o')))
    nc_rx_name, iux = np.unique(np.array([os.path.split(r)[1][:4].lower() for r in nc_list]), return_index=1)
    nc_rx_namea = np.array([os.path.split(r)[1][:4].lower() for r in nc_lista])
    nc_rx_namez = np.array([os.path.split(r)[1][:4].lower() for r in nc_lista])
    
    idn = np.isin(nc_rx_name, rxn)
    fn_list = nc_list[iux][idn]
    nc_list_all = np.empty(fn_list.size, dtype=object)
    for i, nc_rx in enumerate(nc_rx_name):
        tmp = [nc_list[i]]
        ida = np.isin(nc_rx_namea, nc_rx)
        if np.sum(ida) > 0:
            tmp = list(nc_lista[ida]) + tmp 
        idz = np.isin(nc_rx_namez, nc_rx)
        if np.sum(ida) > 0:
            tmp = tmp + list(nc_listz[idz])
        nc_list_all[i] = tmp
    # Nav file
    fsp3 = get_nav_files(navfolder, t)
    # #fsp3 = os.patgation file wasn't found. Sepcified nav_file {navfolder}")
    # Break at the beginning 
    assert not (fsp3==None).all(), "Cant find the sp3 file"
    assert (np.array([os.path.exists(f) for f in fsp3])==1).all(), "Cant find the sp3 file in the directory"
    
    
    # Savename
    sfn = str(year) + '_' + tlim[0].strftime('%m%dT%H%M') + '-' + tlim[1].strftime('%m%dT%H%M') + '_' + os.path.split(rxlist)[1] + '_' + str(el_mask) +'el_' + str(ts) + f's_{int(window_size1)}min_{int(window_size2)}min_{int(window_size3)}min_roti' 
    savefn = os.path.join(odir, sfn + '.h5')
    if not os.path.exists(odir):
        subprocess.call(f'mkdir -p "{odir}"', shell=True)
    
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
    
    if log:
        logfn = os.path.splitext(savefn)[0] + '.log'
        logf = open(logfn, 'w')
        logf.close()
    # Correct tlim for processing purpuses:
    if tlim is not None:
        tlim[0] -= timedelta(hours=1)
        tlim[1] += timedelta(hours=1)
    # This assumes GPS only
    # TODO Update to Galileo
    svl = 32 #gr.load(fnc[0]).sv.values.shape[0]
    rxl = fn_list.size
    # Output arrays
    slanttec = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    residuals_poly = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    residuals_sg1 = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    residuals_sg2 = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    residuals_sg3 = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    roti = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    snr = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    el = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    az = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    
    if log:
        with open(logfn, 'a') as logf:
            logf.write(f"Creating the output file placeholder, {savefn}\n")
        logf.close()
    else:
        print (f"Creating the output file placeholder, {savefn}")
    th5 = gu.datetime2posix(t.astype(datetime))
    h5file = h5py.File(savefn, 'w')
    h5file.create_dataset('obstimes', data=th5)
    h5file.create_dataset('res', data=residuals_poly, compression='gzip', compression_opts=9)
    h5file.create_dataset('res_sg1', data=residuals_sg1, compression='gzip', compression_opts=9)
    h5file.create_dataset('res_sg2', data=residuals_sg2, compression='gzip', compression_opts=9)
    h5file.create_dataset('res_sg3', data=residuals_sg3, compression='gzip', compression_opts=9)
    h5file.create_dataset('roti', data=roti, compression='gzip', compression_opts=9)
    h5file.create_dataset('stec', data=slanttec, compression='gzip', compression_opts=9)
    h5file.create_dataset('snr', data=snr, compression='gzip', compression_opts=9)
    h5file.create_dataset('az', data=az, compression='gzip', compression_opts=9)
    h5file.create_dataset('el', data=el, compression='gzip', compression_opts=9)
    h5file.close()
    if log:
        with open(logfn, 'a') as logf:
            logf.write(f"{savefn} created!\n")
        logf.close()
    else:
        print (f"{savefn} created!")
    del slanttec, residuals_poly, residuals_sg1, residuals_sg2, residuals_sg3, roti, snr, el, az
    
    rxpos = np.nan * np.zeros((rxl, 3), dtype=np.float16)
    rxname = np.zeros(rxl, dtype='<U5')
    rxmodel = np.zeros(rxl, dtype='<U35')
    leap_seconds = 0
    flag = 0
    for irx, fnc in enumerate(nc_list_all):
        ts0 = datetime.now()

        A = do_one(fnc, i=irx, f=savefn, window_size1=window_size1, window_size2=window_size2, window_size3=window_size3)
        if isinstance(A, str):
            flag = 1
        else:
            rxpos[irx,:] = A[0]
            rxname[irx] = A[1]
            rxmodel[irx] = A[2]
            flag = 0
        if log:
            with open(logfn, 'a') as logf:
                if flag == 0:
                    logf.write(f"It took {datetime.now()-ts0} to complete {os.path.split(fnc)[1]}, {irx+1}/{fn_list.size}\n")
                else:
                    logf.write(f"Couldn't process {os.path.split(fnc)[1]}, {irx+1}/{fn_list.size}, {A}\n")
            logf.close()
        else:
            print (f"It took {datetime.now()-ts0} to complete {irx+1}/{fn_list.size}")
    # putting the output file togather
    if log:
        with open(logfn, 'a') as logf:
            logf.write('Saving data...... {}\n'.format(os.path.split(savefn)[1]))
        logf.close()
    else:
        print ('Saving data...... {}'.format(os.path.split(savefn)[1]))
    
    timestamp = datetime.now()
    h5file = h5py.File(savefn, 'r+')
    h5file.create_dataset('rx_positions', data=rxpos, compression='gzip', compression_opts=9)
    asciiListN = [n.encode("ascii", "ignore") for n in rxname]
    h5file.create_dataset('rx_name', (len(asciiListN),1),'S10', asciiListN)
    asciiListM = [n.encode("ascii", "ignore") for n in rxmodel]
    h5file.create_dataset('rx_model', (len(asciiListM),1),'S25', asciiListM)
    h5file.attrs[u'processed'] = timestamp.strftime('%Y-%m-%d')
    h5file.attrs[u'number of receivers'] = rxl
    h5file.attrs[u'el_mask'] = el_mask
    h5file.attrs[u'window_size1'] = window_size1
    h5file.attrs[u'window_size2'] = window_size2
    h5file.attrs[u'window_size3'] = window_size3
    h5file.attrs[u'e0'] = E0
    h5file.attrs[u'leap_seconds'] = leap_seconds
    h5file.close()
    
    if log:
        with open(logfn, 'a') as logf:
            logf.write('{} successfully saved.\n'.format(savefn))
        logf.close()
    else:
        print ('{} successfully saved.'.format(savefn))
    
if __name__ == '__main__':
    global polynom_list, E0, el_mask, ts, use, Hipp
    
    p = ArgumentParser()
    p.add_argument('date')
    p.add_argument('rxlist', type = str, help = 'Rxlist as a .yaml file')
    p.add_argument('odir', help = 'Output filename with or withou root folder.')
    p.add_argument('--elmask', type = int, default = 30)
    p.add_argument('--tlim', default = None, help = "start, stop times example 06:00 08:00", nargs=2, type=str)
    p.add_argument('--obs', help = 'Directory with Rinex files', default=None)
    p.add_argument('--nav', help = 'Directory with Navigation SP3', default=None)
    p.add_argument('--ts', help = 'sampling rate', default = 30, type = int)
    p.add_argument('--window_size1', help = 'Filter window in minutes. Default=30 min', default = 30, type = int)
    p.add_argument('--window_size2', help = 'Filter window in minutes. Default=60 min', default = 60, type = int)
    p.add_argument('--window_size3', help = 'Filter window in minutes. Default=90 min', default = 90, type = int)
    p.add_argument('--e0', help = 'Polyinomial breakout constant; E0 sqrt(30/ts). Default = 0.1', default = 0.1, type = float)
    p.add_argument('--porder', help = 'Number of polynomials. Default=20', default = 20, type = int)
    p.add_argument('--log', help = 'If you prefer to make a .log file?', action = 'store_true')
    P = p.parse_args()
    
    el_mask = P.elmask
    polynom_list = np.arange(0,P.porder)
    ts = P.ts
    E0 = P.e0
    use = 'G'
    Hipp = 450
    
    obsfolder = P.obs if P.obs is not None else os.path.split(P.rxlist)[0] + os.sep
    navfolder = P.nav if P.nav is not None else os.path.split(P.rxlist)[0] + os.sep
    
    main_gps(date=P.date, rxlist=P.rxlist, obsfolder=obsfolder, navfolder=navfolder, 
             window_size1=P.window_size1, window_size2=P.window_size2, window_size3=P.window_size3,
             tlim=P.tlim, odir=P.odir, log=P.log)
