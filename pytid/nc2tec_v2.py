#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:57:09 2019

@author: smrak
"""
from pyGnss import pyGnss, scintillation
from pyGnss import gnssUtils as gu
from datetime import datetime, timedelta
import georinex as gr
import numpy as np
import subprocess
from glob import glob
from dateutil import parser
import yaml
import os
import h5py
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib import dates
import warnings
import platform
warnings.simplefilter('ignore', np.RankWarning)

if platform.system() == 'Linux':
    separator = '/'
else:
    separator = '\\'
    
def getIntervals(y, maxgap=3, maxjump=2):

    r = np.arange(y.size)
    idx = np.isfinite(y)
    r = r[idx]
    intervals=[]
    if len(r)==0:
        return idx, intervals

    beginning=r[0]
    last=r[0]
    for i in r[1:]:
        if (i-last > maxgap) or (abs(y[i] - y[last]) > maxjump):
            intervals.append((beginning, last))
            beginning=i
        last=i
        if i==r[-1]:
            intervals.append((beginning, last))
    return idx, intervals

def plots(dt, stec, elv, tecd_v1, polynom_list, err_list=[], saveroot=None):
    global fnc
    if not isinstance(dt[0], datetime):
        times = np.array([t.astype('datetime64[s]').astype(datetime) for t in dt])
    else:
        times = dt
    fig = plt.figure(figsize=[7,8])
    
    ax0 = fig.add_subplot(311)
    rxname = os.path.split(fnc)[1].split('.')[0][:4]
    ax0.set_title('{} || rxi: {}; svi: {}'.format(rxname, irx, isv))
    ax0.plot(times, stec, 'b')
    ax00 = ax0.twinx()
    ax00.plot(times, elv, '--g', lw=0.5)
    ax0.set_ylabel('STEC')
    ax00.set_ylabel('Elevation', color='g')
    ax00.tick_params(axis='y', colors='green')
    
    ax02 = fig.add_subplot(312, sharex=ax0)
    ax02.plot(times, tecd_v1, 'b')#, label="Sum = {}".format(np.round(np.nansum(tecd_v1)), 3))
    ax02.set_xlabel('time [UT]')
    ax02.set_ylabel('$\Delta$ TEC')
    ax02.grid(axis='y')
    
    ax01 = fig.add_subplot(313)
    ax011 = ax01.twinx()
    try:
        if len(err_list.shape) == 1:
            ax01.semilogy(polynom_list[2:], err_list[2:], '.k')
            ax01.semilogy(polynom_list[2:], err_list[2:], 'k')
            ax011.semilogy(polynom_list[3:], abs(np.diff(err_list))[2:], 'b', )
            ax011.semilogy(polynom_list[3:], abs(np.diff(err_list))[2:], '.b')
            ax011.semilogy([polynom_list[1], polynom_list[-1]], [eps, eps], '--b')
        elif len(err_list.shape) > 1:
            for err_list in err_list:
                ax01.semilogy(polynom_list[2:], err_list[2:], '.k')
                ax01.semilogy(polynom_list[2:], err_list[2:], 'k')
                ax011.semilogy(polynom_list[3:], abs(np.diff(err_list))[2:], 'b', )
                ax011.semilogy(polynom_list[3:], abs(np.diff(err_list))[2:], '.b')
        ax011.semilogy([polynom_list[1], polynom_list[-1]], [eps, eps], '--r', label='E={}'.format(np.round(eps,1)))
        ax011.legend()
        ax01.set_xlabel('Polynomial order')
        ax01.set_ylabel('Error, $|\epsilon |^2$')
        ax011.set_ylabel('$|\Delta \epsilon |^2$', color='blue')
        
        ax011.tick_params(axis='y', colors='blue')
        ax011.grid(axis='y', color='blue')
        ax01.set_xticks(np.arange(0,20,2))
    except:
        pass
    myFmt = dates.DateFormatter('%H:%M')
    ax02.xaxis.set_major_formatter(myFmt)
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax00.get_xticklabels(), visible=False)
    ax02.xaxis.set_major_formatter(myFmt)
    
    plt.tight_layout()
    
    if saveroot is not None:
        if not os.path.exists(saveroot):
            import subprocess
            subprocess.call('mkdir -p {}'.format(saveroot), shell=True, timeout=2)
        prefix = times[0].strftime("%Y%m%d")
        sfn = '{}_rxi{}_svi{}.png'.format(prefix, irx, isv)
        plt.savefig(saveroot + sfn, dpi=100)
        plt.close(fig)
        

def tecdPerLOS(stec, intervals, mask=None, eps=1, polynom_list=None, zero_mean=False):
    tecd = np.nan * np.ones(stec.size)
    if mask is None:
        mask = np.zeros(stec.size, dtype=bool)
    for ir, r in enumerate(intervals):
        chunk = stec[r[0]+1 : r[1]-1]
        idf = np.isfinite(chunk)
        if np.sum(np.isfinite(chunk)) < (15 * (60/tsps)): 
            err_list = np.array([])
            continue
        if np.sum(np.isnan(chunk)) > 0:
            chunk = gu.cubicSplineFit(chunk, idf)
        
        res, err_list0, po  = gu.detrend(chunk, polynom_list=polynom_list, eps=eps, mask=mask[r[0]+1 : r[1]-1], polynomial_order=True)
        polynom_orders.append([(r[1] -r[0]) / (60/tsps), po])
        delta_eps.append(abs(np.diff(err_list0)[-1]))
        if ir == 0 or len(err_list) == 0:
            err_list = err_list0
        else:
            err_list = np.vstack((err_list, err_list0))
        res[~idf] = np.nan
        if zero_mean:
            if abs(np.nansum(res)) < 5:
                tecd[r[0]+1 : r[1]-1] = res
        else:
            tecd[r[0]+1 : r[1]-1] = res
    
    return tecd, err_list

if __name__ == '__main__':
    global fnc, tsps, polynom_orders, delta_eps
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
    p.add_argument('--roti', help = 'compute ROTI?', action = 'store_true')
    p.add_argument('--use_satbias', help = 'Correct the stec for a satbias?', action = 'store_true')
    p.add_argument('--zeromean', help = 'Want to sheck each dtec sector is ~~zero mean?', action = 'store_true')
    p.add_argument('--plot', help = 'Plot the processing steps?', action = 'store_true')
    P = p.parse_args()
    
    # GLOBAL VARIABLES
    if P.cfg is None:
        OBSFOLDER = '/media/smrak/gnss/obs/'
        NAVFOLDER = '/media/smrak/gnss/nav/'
        SBFOLDER = '/media/smrak/gnss/jplg/'
        SAVEFOLDER = '/media/smrak/gnss/hdf/'
        FIGUREFOLDER = '/media/smrak/gnss/plots/'
        
        OBSFOLDER = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\PhD\\dissertation\\python\\data\\obs\\'
        NAVFOLDER = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\PhD\\dissertation\\python\\data\\'
        SBFOLDER = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\PhD\\dissertation\\python\\data\\'
        SAVEFOLDER = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\PhD\\dissertation\\python\\data\\hdf\\'
        FIGUREFOLDER = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\PhD\\dissertation\\python\\data\\testp\\'
    
    else:
        yamlcfg = yaml.load(open(P.cfg, 'r'), Loader=yaml.SafeLoader)
        OBSFOLDER = yamlcfg.get('obsfolder')
        NAVFOLDER = yamlcfg.get('navfolder')
        SBFOLDER = yamlcfg.get('sbfolder')
        SAVEFOLDER = yamlcfg.get('savefolder')
        FIGUREFOLDER = yamlcfg.get('figurefolder')
    date = parser.parse(P.date)
    year = date.year
    doy = date.strftime('%j')
    mmdd = date.strftime('%m%d')
    rxlist = os.path.expanduser(P.rxlist)
    el_mask = P.elmask
    tlim = P.tlim
    Ts = P.ts
    zero_mean = P.zeromean
    
    el_mask_in = (el_mask - 10) if (el_mask - 10) >= 8 else 8
    maxjump = 1.6 + (np.sqrt(Ts) - 1)
    
    PLOT = P.plot
    if PLOT:
        if FIGUREFOLDER is None:
            FIGUREFOLDER = os.path.join(SAVEFOLDER, '{}{}diagnostic{}'.format(mmdd,separator,separator))
    # Obs nav
    nc_root = os.path.join(OBSFOLDER, str(year))
    # Filter input files
    stream = yaml.load(open(rxlist, 'r'), Loader=yaml.FullLoader)
    rxn = np.array(stream.get('rx'))
    # Obs files => Path to
    if os.path.exists(os.path.join(nc_root, str(doy)) + separator):
        pathadd = str(doy) + separator
    else:
        month = str(date.month) if len(str(date.month)) == 2 else '0' + str(date.month)
        day = str(date.day) if len(str(date.day)) == 2 else '0' + str(date.day)
        pathadd = month + day + separator
    nc_folder = os.path.join(nc_root, pathadd)
    assert os.path.exists(nc_folder), "Folder with observation files do not exists."
    nc_list = np.array(sorted(glob(nc_folder + '*.nc')))
    nc_rx_name = np.array([os.path.split(r)[1][:4] for r in nc_list])
    idn = np.isin(nc_rx_name, rxn)
    fnc = nc_list[idn]
    # Nav file
    nav_root = NAVFOLDER
    fnav = os.path.join(nav_root, 'brdc' + str(doy) + '0.' + str(year)[2:] + 'n')
    fsp3 = os.path.join(nav_root, 'igs' + str(doy) + '0.' + str(year)[2:] + 'sp3')
    if not os.path.exists(fnav):
        subprocess.call("rm -rf {}.gz".format(fnav), shell=True)
        dhome = os.path.expanduser("~")
        dldir = os.path.join(dhome, 'pyGnss/utils/download_rnxn.py')
        subprocess.call("python {} {} {} --type gps".format(dldir, P.date, nav_root), shell=True, timeout=50)
    if not os.path.exists(fsp3):
        subprocess.call("rm -rf {}.gz".format(fsp3), shell=True)
        dhome = os.path.expanduser("~")
        dldir = os.path.join(dhome, 'pyGnss/utils/download_rnxn.py')
        subprocess.call("python {} {} {} --type sp3".format(dldir, P.date, nav_root), shell=True, timeout=50)
    # Break at the beginning 
    assert os.path.exists(fsp3), "Cant find the sp3 file"
    
    # jplg file
    if P.use_satbias:
        jplg_root = SBFOLDER
        fjplg = os.path.join(jplg_root, 'jplg' + str(doy) + '0.' + str(year)[2:] + 'i')
        satbias = pyGnss.getSatBias(fjplg)
    #Common time array
    if tlim is None:
        t0 = date #datetime.strptime('{} {}'.format(year,int(doy)),'%Y %j')
        t1 = date + timedelta(days=1) # datetime.strptime('{} {}'.format(year,int(doy) + 1),'%Y %j')
    else:
        assert len(tlim) == 2
        t0 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[0]),'%Y %j-%H:%M')
        t1 = datetime.strptime('{} {}-{}'.format(year,int(doy),tlim[1]),'%Y %j-%H:%M')
    t = np.arange(t0, t1, Ts, dtype='datetime64[s]') #datetime64[s]
    tlim = [t0, t1]
    tl = t.shape[0]
    
    # Savename
    if P.ofn is None:
        sfn = str(year) + '_' + tlim[0].strftime('%m%dT%H%M') + '-' + tlim[1].strftime('%m%dT%H%M') + '_' + os.path.split(rxlist)[1] + '_' + str(el_mask) +'el_' + str(Ts) + 's' 
        if P.roti:
            sfn += '_roti'
        savefn = os.path.join(SAVEFOLDER, sfn + '.h5')
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
    # Polynomial list
    polynom_list = np.arange(0,20)
    # Stats
    # Polynomial orders list for stats
    polynom_orders = []
    delta_eps = []
    # Output arrays
    if P.stec : slanttec = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    residuals = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    if P.roti: roti = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    if Ts == 1: snr = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    el = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    az = np.nan * np.zeros((tl, svl, rxl), dtype=np.float16)
    rxpos = np.nan * np.zeros((rxl, 3), dtype=np.float16)
    rxname = np.zeros(rxl, dtype='<U5')
    rxmodel = np.zeros(rxl, dtype='<U25')
    for irx, fnc in enumerate(fnc):
        # New Array
        try:
            svlist = gr.load(fnc).sv.values
            leap_seconds = gu.getLeapSeconds(fnav)
            D = gr.load(fnc)
            dt = np.array([np.datetime64(ttt) for ttt in D.time.values]).astype('datetime64[s]').astype(datetime) - timedelta(seconds=leap_seconds)
            tsps = np.diff(dt.astype('datetime64[s]'))[0].astype(int)
            eps = 1 * np.sqrt(30/tsps)
            VTEC, F, AER = pyGnss.getVTEC(fnc=fnc, fsp3=fsp3, jplg_file=None,
                                     el_mask=el_mask_in, 
                                     return_mapping_function=True,
                                     return_aer=True, maxgap=1, maxjump=maxjump)
            if Ts == 1: 
                SNR = pyGnss.getCNR(D, fsp3=fsp3, el_mask=el_mask, H=350)
            # Remove inital recovery at time 00:00
            VTEC[:2,:] = np.nan
            try:
                rxmodel[irx] = gr.load(fnc).rxmodel
            except:
                pass
            rxpos[irx] = gr.load(fnc).position_geodetic
            rxname[irx] = gr.load(fnc).filename[:4]
            if P.log:
                with open(logfn, 'a') as LOG:
                    LOG.write('Processing {}/{}\n'.format(irx+1, rxl))
                    LOG.close()
            else:
                print ('{}/{}'.format(irx+1, rxl))
            for isv, sv in enumerate(svlist):
                try:
                    if isv > 32: 
                        continue
                    ixmask = np.nan_to_num(AER[:, isv, 1]) >= el_mask
                    
                    idx, intervals = getIntervals(VTEC[:, isv], maxgap=1, maxjump=maxjump)
                    tecd, err_list = tecdPerLOS(VTEC[:, isv], intervals, polynom_list=polynom_list, eps=eps)
                    tecd[~ixmask] = np.nan
                    tec = VTEC[:, isv]
                    tec[~ixmask] = np.nan
                    
                    idt = np.isin(t, dt[ixmask])
                    idt_reverse = np.isin(dt[ixmask], t[idt])
                    
                    # Store to output arrays
                    residuals[idt, isv, irx] = tecd[ixmask][idt_reverse]
                    el[idt, isv, irx] = AER[:, isv, 1][ixmask][idt_reverse]
                    az[idt, isv, irx] = AER[:, isv, 0][ixmask][idt_reverse]
                    
                    # Optionals
                    if P.stec: 
                        slanttec[idt, isv, irx] = VTEC[:, isv][ixmask][idt_reverse]
                    
                    if P.roti:
                        rot = np.hstack((np.nan, (np.diff(tec) / tsps)))
                        if Ts == 1:
                            roti_temp = scintillation.sigmaTEC(rot, 60)
                        else: #5 min
                            roti_T = int((60*5) / tsps)
                            roti_temp = scintillation.sigmaTEC(rot, roti_T)
                        roti[idt, isv, irx] = roti_temp[ixmask][idt_reverse]
        
                    if Ts == 1: 
                        try:
                            S1 = SNR[:, isv]
                        except:
                            S1 = np.nan * np.ones(dt.size)
                        snr[idt, isv, irx] = S1[ixmask][idt_reverse]
                    # Plot
                    if PLOT:
                        plots(dt, tec, AER[:, isv, 1], tecd, polynom_list, err_list, saveroot=FIGUREFOLDER)
                except Exception as e:
                    print ("Skipped: Rx: {}, SV:{}".format(irx, isv))
                    print (e)
    #                    if P.log:
    #                        LOG.write(str(e) + '\n')
                    
    #                        pass
    #                    else:
    #                        print (e)
        except Exception as e:
            print (e)
#            if P.log:
#                with open(logfn, 'a') as LOG:
#                    LOG.write(str(e) + '\n')
#                LOG.close()
#            else:
#                print (e)
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
    if P.roti:
        h5file.create_dataset('roti', data=roti, compression='gzip', compression_opts=9)
    if P.stec:
        h5file.create_dataset('stec', data=slanttec, compression='gzip', compression_opts=9)
    if Ts == 1: 
        h5file.create_dataset('snr', data=snr, compression='gzip', compression_opts=9)
    h5file.create_dataset('az', data=az, compression='gzip', compression_opts=9)
    h5file.create_dataset('el', data=el, compression='gzip', compression_opts=9)
    h5file.create_dataset('rx_positions', data=rxpos, compression='gzip', compression_opts=9)
    try:
        asciiListN = [n.encode("ascii", "ignore") for n in rxname]
        h5file.create_dataset('rx_name', (len(asciiListN),1),'S10', asciiListN)
        asciiListM = [n.encode("ascii", "ignore") for n in rxmodel]
        h5file.create_dataset('rx_model', (len(asciiListM),1),'S25', asciiListM)
    except:
        pass
    timestamp = datetime.now()
    h5file.attrs[u'processed'] = timestamp.strftime('%Y-%m-%d')
    h5file.attrs[u'number of receivers'] = rxl
    h5file.attrs[u'el_mask'] = el_mask
    # Stats
    po_length = np.array(polynom_orders)[:,0]
    po = np.array(polynom_orders)[:,1]
    h5file.create_dataset('po', data=po, compression='gzip', compression_opts=9)
    h5file.create_dataset('po_length', data=po_length, compression='gzip', compression_opts=9)
    h5file.create_dataset('delta_eps', data=np.array(delta_eps), compression='gzip', compression_opts=9)
    # Close the HDF5
    h5file.close()
    if P.log:
        with open(logfn, 'a') as LOG:
            LOG.write('{} successfully saved.\n'.format(savefn))
        LOG.close()
    else:
        print ('{} successfully saved.'.format(savefn))
