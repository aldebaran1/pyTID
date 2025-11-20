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
from dateutil import parser
import pyarrow as pa
import yaml
import os
import xarray as xr
import h5py
import astropy.time as at
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

def do_one(fnc, use, roti_cutoff=0.5, snr_cutoff=30, Hipp=450):
    # try:
    sfn = odir+os.sep+os.path.split(fnc[0])[1][:4].lower()+".nc"
    # if os.path.exists(sfn):
    #      return
    try:
        for ifn, fn in enumerate(fnc):
            if ifn == 0:
                D = gr.load(fn, use = use, fast=1, interval=ts, tlim=(t[0].astype(datetime), t[-1].astype(datetime)))
            else:
                D = xr.concat((D, gr.load(fn, use = use, fast=2, interval=ts, tlim=(t[0].astype(datetime), t[-1].astype(datetime)))), dim='time')
        tsps = D.interval
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
            "GPS_Leap_Seconds": int(leap_seconds),
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
        O.attrs = attrs
        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in O.data_vars}
        O.to_netcdf(sfn, mode='w', encoding=encoding)
        
    except Exception as e:
        with open(sfn+'.txt') as f:
            f.write(str(e))
    
    del D, O
    
    return
    
# except Exception as e:
    #     print (e)
    
    # pass


date = parser.parse('2020-12-27')
year = date.year
yeara = (date - timedelta(days=1)).year
yearz = (date + timedelta(days=1)).year
doy = date.strftime('%j')
mmdd = date.strftime('%m%d')
mmdda = (date - timedelta(days=1)).strftime('%m%d')
mmddz = (date + timedelta(days=1)).strftime('%m%d')

root = '/Users/mraks1/Library/CloudStorage/Box-Box/Projects/TID-GW/RINEX/'
obsfolder = root
navfolder = root 
odir = '/Volumes/SD-1/test/'
rxlist = obsfolder + f'{os.sep}{year}{os.sep}{mmdd}{os.sep}rxlist1227.2020.h5.yaml'
tlim = None
ts = 30
window_size0 = 5
window_size1 = 30
window_size2 = 60
window_size3 = 90
el_mask = 20
E0 = 0.1
polynom_list = np.arange(20)
log = 0
year = date.year
doy = date.strftime('%j')
use = ('G', 'E', 'C')
# use = 'E'

# Filter input files
stream = yaml.safe_load(open(rxlist, 'r'))
rxn = np.array(stream.get('rx'))
# Obs files => Path to
assert os.path.exists(obsfolder), "Folder with observation files does not exists."
nc_list = np.array(sorted(glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' +  '*crx') + glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' + '*rnx') + glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' + '*.*d') + glob(obsfolder + f'{year}{os.sep}{mmdd}{os.sep}' + '*.*o')))
nc_lista = np.array(sorted(glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' +  '*crx') + glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*rnx') + glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*.*d') + glob(obsfolder + f'{yeara}{os.sep}{mmdda}{os.sep}' + '*.*o')))
nc_listz = np.array(sorted(glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' +  '*crx') + glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*rnx') + glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*.*d') + glob(obsfolder + f'{yearz}{os.sep}{mmddz}{os.sep}' + '*.*o')))
nc_rx_name, iux = np.unique([os.path.split(r)[1][:4] for r in nc_list], return_index=1)
nc_rx_namea, iua = np.unique([os.path.split(r)[1][:4] for r in nc_lista], return_index=1)
nc_rx_namez, iuz = np.unique([os.path.split(r)[1][:4] for r in nc_listz], return_index=1)

idn = np.isin([n.lower() for n in nc_rx_name], rxn)
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

#Common time array
if tlim is None:
    delta_t = timedelta(hours=3)
    t0 = date 
    t1 = date + timedelta(days=1)
else:
    assert len(tlim) == 2
    t0 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[0]),'%Y %j-%H:%M')
    t1 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[1]),'%Y %j-%H:%M')
    delta_t = timedelta(seconds=0)
t = np.arange(t0-delta_t, t1+delta_t, ts, dtype='datetime64[s]') #datetime64[s]
# tt = np.arange(t0, t1, ts, dtype='datetime64[s]') #datetime64[s]
# tlim = [t0, t1]
tl = t.size

# Nav file
fsp3 = get_nav_files(navfolder, t)
# Break at the beginning 
assert not (fsp3==None).all(), "Cant find the sp3 file"
assert (np.array([os.path.exists(f) for f in fsp3])==1).all(), "Cant find the sp3 file in the directory"
    
# Savename
# sfn = str(year) + '_' + t0.strftime('%m%dT%H%M') + '-' + t1.strftime('%m%dT%H%M') + '_' + os.path.split(rxlist)[1] + '_' + str(el_mask) +'el_' + str(ts) + 's_ra_roti' 
# savefn = os.path.join(odir, sfn + '.nc')
# if not os.path.exists(odir):
#     subprocess.call(f'mkdir -p "{odir}"', shell=True)

# # Dealing with duplicate file names
# if os.path.exists(savefn):
#     head = os.path.splitext(savefn)[0]
#     c = 0
#     while os.path.exists(savefn):
#         try:
#             c = int(os.path.splitext(savefn)[0].split('_')[-1])
#             c += 1
#         except:
#             c += 1
#         savefn = head + '_' + str(c) + '.nc'

# if log:
#     logfn = os.path.splitext(savefn)[0] + '.log'
#     logf = open(logfn, 'w')
#     logf.close()

leap_seconds = 18
flag = 0
for irx, fnc in enumerate(nc_list_all):
    # fnc=nc_list_all[2]
    ts0 = datetime.now()
    do_one(fnc, use=use)
    print (f"It took {datetime.now()-ts0} to complete {irx+1}/{fn_list.size}")
    # break
    # if log:
    #     with open(logfn, 'a') as logf:
    #         if flag == 0:
    #             logf.write(f"It took {datetime.now()-ts0} to complete {os.path.split(fnc)[1]}, {irx+1}/{fn_list.size}\n")
    #         else:
    #             logf.write(f"Couldn't process {os.path.split(fnc)[1]}, {irx+1}/{fn_list.size}\n")
    #     logf.close()
    # else:
    #     print (f"It took {datetime.now()-ts0} to complete {irx+1}/{fn_list.size}")
    if irx > 10:
        break
    
# putting the output file togather
# if log:
#     with open(logfn, 'a') as logf:
#         logf.write('Saving data...... {}\n'.format(os.path.split(savefn)[1]))
#     logf.close()
# else:
#     print ('Saving data...... {}'.format(os.path.split(savefn)[1]))

# timestamp = datetime.now()
# h5file = h5py.File(savefn, 'r+')
# h5file.create_dataset('rx_positions', data=rxpos, compression='gzip', compression_opts=9)
# asciiListN = [n.encode("ascii", "ignore") for n in rxname]
# h5file.create_dataset('rx_name', (len(asciiListN),1),'S10', asciiListN)
# asciiListM = [n.encode("ascii", "ignore") for n in rxmodel]
# h5file.create_dataset('rx_model', (len(asciiListM),1),'S25', asciiListM)
# h5file.attrs[u'processed'] = timestamp.strftime('%Y-%m-%d')
# h5file.attrs[u'number of receivers'] = rxl
# h5file.attrs[u'el_mask'] = el_mask
# h5file.attrs[u'window_size'] = window_size
# h5file.attrs[u'e0'] = E0
# try:
#     h5file.attrs[u'leap_seconds'] = leap_seconds
# except:
#     pass
# h5file.close()

# if log:
#     with open(logfn, 'a') as logf:
#         logf.write('{} successfully saved.\n'.format(savefn))
#     logf.close()
# else:
#     print ('{} successfully processed.'.format(savefn))
    

