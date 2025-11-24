#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:57:09 2019

@author: smrak
"""
from pyGnss import pyGnss
import multiprocessing as mp
from pyGnss import gnssUtils as gu
from datetime import datetime, timedelta
import georinex as gr
import numpy as np
import subprocess
from glob import glob
from dateutil import parser
import yaml
import os
import xarray as xr
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

roti_cutoff = 0.5
snr_cutoff = 30
Hipp = 450
window_size0 = 5
window_size1 = 30
window_size2 = 60
window_size3 = 90

use = ('G', 'E', 'C')

def get_nav_files(navfolder, times):
    if not isinstance(times.dtype, datetime):
        times = times.astype('datetime64[s]').astype(datetime)
    days = np.unique([date.date() for date in times]).astype('datetime64[s]').astype(datetime)
    for i, day in enumerate(days):
        # d = day.day
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

def do_one(fnc, fsp3, t, ts, odir, el_mask=30,
           # roti_cutoff=0.5, snr_cutoff=30, Hipp=450, use='G',
           # window_size0=5, window_size1=30, window_size2=60, window_size3=90
           ):
    sfn = f"{odir}{os.path.split(fnc[0])[1][:4].lower()}_{t[0].astype(datetime).strftime('%Y%m%dT%H%M')}_{t[-1].astype(datetime).strftime('%Y%m%dT%H%M')}.nc"
    if os.path.exists(sfn):
         return
    try:
        t0 = datetime.now()
        for ifn, fn in enumerate(fnc):
            if ifn == 0:
                tmp = gr.load(fn, use = use, fast=1, interval=ts, tlim=(t[0].astype(datetime), t[-1].astype(datetime)))
                if tmp.sizes["time"] > 0:
                    D = tmp
            else:
                tmp = gr.load(fn, use = use, fast=1, interval=ts, tlim=(t[0].astype(datetime), t[-1].astype(datetime)))
                if tmp.sizes["time"] > 0:
                    if 'D' in locals():
                        D = xr.concat((D, tmp), dim='time')
                    else:
                        D = tmp
        if 'D' not in locals():
            return
        # print (f"It took {datetime.now()-t0} to load RINEX")
        tsps = D.interval
        if np.isnan(tsps):
            tsps = int(np.nanmedian(np.diff(D.time)) / np.timedelta64(1, 's'))
        maxjump = 1.6 + (np.sqrt(tsps) - 1)
        N0 = int((60/tsps)*window_size0)
        N1 = int((60/tsps)*window_size1)
        N2 = int((60/tsps)*window_size2)
        N3 = int((60/tsps)*window_size3)
        NROTI = int((60/tsps) * 5) # ROTI over 5 min
        # eps = E0 * np.sqrt(30/tsps)
        # TODO Correct GPST too UTC time when calling AER in pyGnss
        STEC, TEC_sigma, AER = pyGnss.getSTEC(fnc=D, fsp3=fsp3, el_mask=el_mask,
                                   maxgap=1, maxjump=maxjump, return_aer=1,
                                   return_tec_error=1,
                                   )
        ROTI = pyGnss.getROTI(STEC, ts=tsps, N = NROTI)
        SNR = pyGnss.getCNR(D, fsp3=fsp3, el_mask=el_mask, H=Hipp)
        DCB = pyGnss.getDCBfromSTEC(STEC, AER, el_mask=5, ROTI=ROTI, roti_cutoff=roti_cutoff, SNR=SNR, snr_cutoff=roti_cutoff)
        F = pyGnss.getMappingFunction(AER[:,:,1], Hipp)
        STECcorr = STEC - DCB
        VTEC = STECcorr * F
        VTEC[TEC_sigma > 8] = np.nan
        D['stec'] = (("time", "sv"), STECcorr)
        O = D['stec'].to_dataset()
        O['tec_sigma'] = (("time", "sv"), TEC_sigma)
        O['snr'] = (("time", "sv"), SNR)
        O['roti'] = (("time", "sv"), ROTI)
        O['el'] = (("time", "sv"), AER[:,:,1])
        O['az'] = (("time", "sv"), AER[:,:,0])
        O['dtec0'] = (("time", "sv"), pyGnss.getDTECsg_from_VTEC(VTEC, N=N0, order=1))
        O['dtec1'] = (("time", "sv"), pyGnss.getDTECsg_from_VTEC(VTEC, N=N1, order=1))
        O['dtec2'] = (("time", "sv"), pyGnss.getDTECsg_from_VTEC(VTEC, N=N2, order=1))
        O['dtec3'] = (("time", "sv"), pyGnss.getDTECsg_from_VTEC(VTEC, N=N3, order=1))
        O = O.stack(flat=["time", "sv"]).reset_index("flat").dropna(dim="flat", subset=["stec"]).unstack()
        O['time'] = ("flat", O.time.values)
        O['sv'] = ("flat", O.sv.values)
        
        attrs={"title": "TEC Data",
            "position_geodetic": D.position_geodetic,
            "filename": D.filename,
            "ReferenceFrame": "WGS84",
            "svall": D.sv.values,
            "obstimes": gu.datetime2posix(D.time.values.astype('datetime64[s]').astype(datetime)),
            "satellite_type": use,
            "DataType": use,
            "DateSubmitted": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            "DataInterval": ts,
            "Elevation mask": el_mask,
            "ROTI Cutoff": roti_cutoff,
            "Signal Strength Cutoff": snr_cutoff,
            "hipp": Hipp,
            "software": "nc2rnx",
            "author": "sm"
            }
        
        if 'rxmodel' in D.attrs:
            attrs['rxmodel'] = D.attrs['rxmodel']
        if 'leap_seconds' in D.attrs:
            attrs["GPS_Leap_Seconds"] = D.leap_seconds,
        O.attrs = attrs
        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in O.data_vars}
        O.to_netcdf(sfn, mode='w', encoding=encoding)
        del D, O
    except Exception as e:
        with open(sfn+'.txt', 'w') as f:
            f.write(str(e))
    
    return
    
def main(date, idir, ndir, odir, rxlist, el_mask, tlim, ts, j, v=False):
    date = parser.parse(date)
    year = date.year
    yeara = (date - timedelta(days=1)).year
    yearz = (date + timedelta(days=1)).year
    doy = date.strftime('%j')
    mmdd = date.strftime('%m%d')
    mmdda = (date - timedelta(days=1)).strftime('%m%d')
    mmddz = (date + timedelta(days=1)).strftime('%m%d')
    
    year = date.year
    doy = date.strftime('%j')
    
    assert os.path.exists(idir), "Folder with observation files does not exists."
    nc_list = np.array(sorted(glob(idir + f'{year}{os.sep}{mmdd}{os.sep}' +  '*crx') + glob(idir + f'{year}{os.sep}{mmdd}{os.sep}' + '*rnx') + glob(idir + f'{year}{os.sep}{mmdd}{os.sep}' + '*.*d') + glob(idir + f'{year}{os.sep}{mmdd}{os.sep}' + '*.*o')))
    nc_lista = np.array(sorted(glob(idir + f'{yeara}{os.sep}{mmdda}{os.sep}' +  '*crx') + glob(idir + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*rnx') + glob(idir + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*.*d') + glob(idir + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*.*o')))
    nc_listz = np.array(sorted(glob(idir + f'{yearz}{os.sep}{mmddz}{os.sep}' +  '*crx') + glob(idir + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*rnx') + glob(idir + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*.*d') + glob(idir + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*.*o')))
    nc_rx_name, iux = np.unique([os.path.split(r)[1][:4] for r in nc_list], return_index=1)
    nc_rx_namea, iua = np.unique([os.path.split(r)[1][:4] for r in nc_lista], return_index=1)
    nc_rx_namez, iuz = np.unique([os.path.split(r)[1][:4] for r in nc_listz], return_index=1)
    
    if rxlist is not None:
        stream = yaml.safe_load(open(rxlist, 'r'))
        rxn = np.array(stream.get('rx'))
        idn = np.isin([n.lower() for n in nc_rx_name], rxn)
    else:
        idn = np.ones(nc_rx_name.size, dtype=bool)
    
    fn_list = nc_list[iux][idn]
    nc_list_all = np.empty(fn_list.size, dtype=object)
    for i, nc_rx in enumerate(nc_rx_name):
        tmp = [nc_list[np.isin(nc_rx_name, nc_rx)].item()]
        ida = np.isin(nc_rx_namea, nc_rx)
        if np.sum(ida) > 0:
            tmp = list(nc_lista[iua][ida]) + tmp
        idz = np.isin(nc_rx_namez, nc_rx)
        if np.sum(idz) > 0:
            tmp = tmp + list(nc_listz[iuz][idz])
        nc_list_all[i] = tmp
    
    if tlim is None:
        delta_t = timedelta(hours=3, minutes=1)
        t0 = date 
        t1 = date + timedelta(days=1)
    else:
        assert len(tlim) == 2
        t0 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[0]),'%Y %j-%H:%M')
        t1 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[1]),'%Y %j-%H:%M')
        delta_t = timedelta(seconds=0)
    t = np.arange(t0-delta_t, t1+delta_t, ts, dtype='datetime64[s]') #datetime64[s]
    
    fsp3 = get_nav_files(ndir, t)
    # Break at the beginning 
    assert not (fsp3==None).all(), "Cant find the sp3 file"
    assert (np.array([os.path.exists(f) for f in fsp3])==1).all(), "Cant find the sp3 file in the directory"
    # exit()
    args = []
    for i in range(nc_list_all.size):
        args.append((nc_list_all[i], fsp3, t, ts, odir, el_mask))
    t0 = datetime.now()
    
    odir += f"{mmdd}{os.sep}" 
    if not os.path.exists(odir):
        subprocess.call(f"mkdir -p {odir}", shell=True)
    with mp.Pool(processes=j) as pool:
            # Map the square_number function to each number in the list
            # The pool will distribute these calls across its processes
            pool.starmap(do_one, args)
    if v:
        print (f"It took {datetime.now()-t0} to process {nc_list_all.size} files with 8 parallel processes")
    
if __name__ == "__main__":
    
    p = ArgumentParser()
    p.add_argument('date', help = 'Root Rinex Obs Dirctory')
    p.add_argument('idir', help = 'Root Rinex Obs Dirctory')
    p.add_argument('ndir', help = 'Rinex Navigation files Dirctory')
    p.add_argument('odir', help = 'Output filename with or withou root folder.')
    
    p.add_argument('--rxlist', type = str, help = 'Rxlist as a .yaml file', default=None)
    
    p.add_argument('--elmask', help="Elevation mask. Default=10", type = int, default = 10)
    p.add_argument('--tlim', default = None, help = "start, stop times example 06:00 08:00", nargs=2, type=str)
    
    p.add_argument('--ts', help = 'Sampling rate. Default=30', default = 30, type = int)
    p.add_argument('-j', help = 'Number of parallel processes. Default=2', type=int, default=2)
    p.add_argument('-v', help = 'Verbose', action = 'store_true')
    P = p.parse_args()
    
    main(date=P.date, idir=P.idir, ndir=P.ndir, odir=P.odir, rxlist=P.rxlist, 
         tlim=P.tlim, el_mask=P.elmask, ts=P.ts, j=P.j, v=P.v)


    # root = '/Users/mraks1/Library/CloudStorage/Box-Box/Projects/TID-GW/RINEX/'
    # obsfolder = root
    # navfolder = root 
    # odir = '/Volumes/SD-1/test/'
    # rxlist = obsfolder + f'{os.sep}{year}{os.sep}{mmdd}{os.sep}rxlist1227.2020.h5.yaml'
    # tlim = None
    # ts = 30
    
    # el_mask = 20
    # E0 = 0.1
    # polynom_list = np.arange(20)
    # log = 0
    
    
    # Filter input files
    # stream = yaml.safe_load(open(rxlist, 'r'))
    # rxn = np.array(stream.get('rx'))
    # Obs files => Path to
    # assert os.path.exists(obsfolder), "Folder with observation files does not exists."
    # nc_list = np.array(sorted(glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' +  '*crx') + glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' + '*rnx') + glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' + '*.*d') + glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' + '*.*o')))
    # nc_lista = np.array(sorted(glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' +  '*crx') + glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*rnx') + glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*.*d') + glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*.*o')))
    # nc_listz = np.array(sorted(glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' +  '*crx') + glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*rnx') + glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*.*d') + glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*.*o')))
    # nc_rx_name, iux = np.unique([os.path.split(r)[1][:4] for r in nc_list], return_index=1)
    # nc_rx_namea, iua = np.unique([os.path.split(r)[1][:4] for r in nc_lista], return_index=1)
    # nc_rx_namez, iuz = np.unique([os.path.split(r)[1][:4] for r in nc_listz], return_index=1)
    
    # idn = np.isin([n.lower() for n in nc_rx_name], rxn)
    # fn_list = nc_list[iux][idn]
    # nc_list_all = np.empty(fn_list.size, dtype=object)
    # for i, nc_rx in enumerate(nc_rx_name):
    #     tmp = [nc_list[np.isin(nc_rx_name, nc_rx)].item()]
    #     ida = np.isin(nc_rx_namea, nc_rx)
    #     if np.sum(ida) > 0:
    #         tmp = list(nc_lista[iua][ida]) + tmp
    #     idz = np.isin(nc_rx_namez, nc_rx)
    #     if np.sum(idz) > 0:
    #         tmp = tmp + list(nc_listz[iuz][idz])
    #     nc_list_all[i] = tmp
    
    #Common time array
    # if tlim is None:
    #     delta_t = timedelta(hours=3)
    #     t0 = date 
    #     t1 = date + timedelta(days=1)
    # else:
    #     assert len(tlim) == 2
    #     t0 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[0]),'%Y %j-%H:%M')
    #     t1 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[1]),'%Y %j-%H:%M')
    #     delta_t = timedelta(seconds=0)
    # t = np.arange(t0-delta_t, t1+delta_t, ts, dtype='datetime64[s]') #datetime64[s]
    # tl = t.size
    # Nav file
    # fsp3 = get_nav_files(navfolder, t)
    # # Break at the beginning 
    # assert not (fsp3==None).all(), "Cant find the sp3 file"
    # assert (np.array([os.path.exists(f) for f in fsp3])==1).all(), "Cant find the sp3 file in the directory"
    # # exit()
    # args = []
    # for i in range(nc_list_all.size):
    #     args.append((nc_list_all[i], fsp3, t, ts, odir, use, roti_cutoff, snr_cutoff, Hipp, el_mask, window_size0, window_size1, window_size2, window_size3))
    # t0 = datetime.now()
    # with mp.Pool(processes=j) as pool:
    #         # Map the square_number function to each number in the list
    #         # The pool will distribute these calls across its processes
    #         pool.starmap(do_one, args)
    # print (f"It took {datetime.now()-t0} to process {nc_list_all.size} files with 8 parallel processes")
    