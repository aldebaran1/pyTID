# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:06:16 2020

@author: smrak@bu.edu
"""

import numpy as np
import os, h5py
from pyGnss import scintillation
from pyGnss import pyGnss
from datetime import datetime
import warnings
from argparse import ArgumentParser

warnings.simplefilter('ignore', np.RankWarning)

global tsps

def main(F, el_mask = None, odir = None):
    global tsps
    D = h5py.File(F, 'r')
    el_mask = 20
    maxjump = 1.6 + (np.sqrt(tsps) - 1)
    eps = 1 * np.sqrt(30/tsps)
    polynom_list = np.arange(0,20)

    el_mask_in = (el_mask - 10) if (el_mask - 10) >= 8 else 8
    
    print ('Reading in receiver names')
    t0 = datetime.now()
    obstimes_all = []
    rxn_all = []
    sv_all = []
    for row in D['Data/Table Layout'][()]:
        obstimes_all.append(row[9])
        rxn_all.append(row[12].decode())
        sv_all.append(row[13])
    
    obstimes_unix = np.unique(np.asarray(obstimes_all))
    rxn_all = np.asarray(rxn_all)
    sv_all = np.asarray(sv_all)
    print ('It took {}, to read: times, satelites and receivers'.format(datetime.now()-t0))
    D.close()
    
    rxn = np.unique(rxn_all)
    sv_unique = np.unique(sv_all)
    sv_index = np.arange(sv_unique.size)
    sv_list = {}
    for i, s in enumerate(sv_unique):
        sv_list[str(s)] = sv_index[i]


    # Out-filename
    t_start = datetime.utcfromtimestamp(obstimes_unix[0])
    t_end = datetime.utcfromtimestamp(obstimes_unix[-1])
    if odir is None:
        odir = os.path.split(F)[0]
    sfn = str(t_start.year) + '_' + t_start.strftime('%m%dT%H%M') + '-' + t_end.strftime('%m%dT%H%M') + '_' + 'madrigallos' + '_' + str(el_mask) +'el_' + str(tsps) + 's' + '_roti'
    savefn = os.path.join(odir, sfn + '.h5')
    # Duplicate file names
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
        
    logfn = os.path.splitext(savefn)[0] + '.log'
    LOG = open(logfn, 'w')
    LOG.close()
    print ('Init arrays')
    TEC = np.nan * np.ones((obstimes_unix.size, sv_unique.size, rxn.size), dtype=np.float16)
    DTEC = np.nan * np.ones((obstimes_unix.size, sv_unique.size, rxn.size), dtype=np.float16)
    ROTI = np.nan * np.ones((obstimes_unix.size, sv_unique.size, rxn.size), dtype=np.float16)
    AZ = np.nan * np.ones((obstimes_unix.size, sv_unique.size, rxn.size), dtype=np.float16)
    EL = np.nan * np.ones((obstimes_unix.size, sv_unique.size, rxn.size), dtype=np.float16) 
    RXP = np.nan * np.ones((rxn.size, 3), dtype=np.float16)
    print ('Saving to: {}'.format(savefn))
    h5file = h5py.File(savefn, 'w')
    h5file.create_dataset('obstimes', data=obstimes_unix)
    h5file.create_dataset('vtec', data=TEC, compression='gzip', compression_opts=9)
    h5file.create_dataset('res', data=DTEC, compression='gzip', compression_opts=9)
    h5file.create_dataset('roti', data=ROTI, compression='gzip', compression_opts=9)
    h5file.create_dataset('az', data=AZ, compression='gzip', compression_opts=9)
    h5file.create_dataset('el', data=EL, compression='gzip', compression_opts=9)
    h5file.create_dataset('rx_positions', data=RXP, compression='gzip', compression_opts=9)
    asciiListN = [n.encode("ascii", "ignore") for n in rxn]
    h5file.create_dataset('rx_name', (len(asciiListN),1),'S4', asciiListN)
    h5file.close()

    del TEC, DTEC, ROTI, AZ, EL, RXP
    print ('Arrays erased')
    for irx, rx in enumerate(rxn):
        
        if irx == 0:
            with open(logfn, 'a') as LOG:
                LOG.write('Processing {}/{}.\n'.format(irx+1, rxn.size))
                LOG.close()
        else:
            with open(logfn, 'a') as LOG:
                LOG.write('Processing {}/{}. It took {} to process last RX data. \n'.format(irx+1, rxn.size, datetime.now()-t0))
                LOG.close()
        t0 = datetime.now()
        
        try:
            D = h5py.File(F, 'r')
            idrx = np.isin(rxn_all, rx)
            svn = np.unique(sv_all[idrx])
            rx_lat = D['Data/Table Layout'][idrx][0][-4]
            rx_lon = D['Data/Table Layout'][idrx][0][-3]
            
            t = []
            vtec_tmp = []
            elv_tmp = []
            azm_tmp = []
            for row in D['Data/Table Layout'][idrx]:
                t.append(row[9])
                vtec_tmp.append(row[18])
                elv_tmp.append(row[-5])
                azm_tmp.append(row[-6])
            t = np.array(t)
            vtec_tmp = np.array(vtec_tmp)
            elv_tmp = np.array(elv_tmp)
            azm_tmp = np.array(azm_tmp)
            D.close()
            
            h5file = h5py.File(savefn, 'a')
            h5file['rx_positions'][irx, 0] = rx_lat
            h5file['rx_positions'][irx, 1] = rx_lon
            h5file['rx_positions'][irx, 2] = 0
            h5file.close()
            
            del rx_lat, rx_lon
            
            for isv, sv in enumerate(svn):
                vtec = np.nan * np.ones(obstimes_unix.size, dtype=np.float16)
                elv = np.nan * np.ones(obstimes_unix.size, dtype=np.float16)
                azm = np.nan * np.ones(obstimes_unix.size, dtype=np.float16)
                roti = np.nan * np.ones(obstimes_unix.size, dtype=np.float16)
                h5file = h5py.File(savefn, 'a')
                
                idsv = np.isin(sv_all[idrx], sv)
                ids = sv_list[str(sv)]
                
                idt = np.isin(obstimes_unix, t[idsv])
                vtec[idt] = vtec_tmp[idsv]
                azm[idt] = azm_tmp[idsv]
                elv[idt] = elv_tmp[idsv]
                del idt
                
                idel0 = np.nan_to_num(elv) < el_mask_in
                idel = np.nan_to_num(elv) < el_mask
                vtec[idel0] = np.nan
                try:
                    idx, intervals = pyGnss.getIntervalsTEC(vtec, maxgap=5, maxjump=maxjump)
                    tecd = pyGnss.tecdPerLOS(vtec, intervals, polynom_list=polynom_list, eps=eps)
                    tecd[idel] = np.nan
                    h5file['res'][:, ids, irx] = tecd
                    del tecd
                except Exception as e:
                    print (e)
                try:
                    vtec[idel] = np.nan
                    rot = np.hstack((np.nan, (np.diff(vtec) / tsps)))
                    roti = scintillation.sigmaTEC(rot, 10) # 5 min
                    
                    h5file['vtec'][:, ids, irx] = vtec
                    h5file['roti'][:, ids, irx] = roti
                    h5file['el'][:, ids, irx] = elv
                    h5file['az'][:, ids, irx] = azm
                    
                    del idx, intervals, idel0, idel
                    
                except Exception as e:
                    print (e)
                
                h5file.close()
                
            del vtec, roti, elv, azm
        except Exception as e:
            print (e)
        

    with open(logfn, 'a') as LOG:
        LOG.write('Processing Done')
        LOG.close()
        
    return 0
    
if __name__ == '__main__':
    tsps =  30
    p = ArgumentParser()
    p.add_argument('filename', help= 'to madrigal_los.hdf5')
    p.add_argument('--elmask', type = int, default = 30)
    p.add_argument('--odir', help = 'Output directory.', default=None)
    P = p.parse_args()
    
    main(P.filename, el_mask=P.elmask, odir=P.odir)