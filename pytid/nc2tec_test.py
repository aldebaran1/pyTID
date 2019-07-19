# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:36:42 2019

@author: smrak@bu.edu
"""
import os
import georinex as gr
from pyGnss import pyGnss
from pyGnss import gnssUtils as gu
import numpy as np
from glob import glob
from datetime import datetime
from matplotlib import dates
import matplotlib.pyplot as plt
import h5py

def plots(dt, stec, elv, tecd_v1, polynom_list, err_list, odir=None, PLOT=1):
    times = np.array([t.astype('datetime64[s]').astype(datetime) for t in dt])
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
    if len(err_list.shape) == 1:
        ax01.semilogy(polynom_list[2:], abs(err_list[2:]), '.k')
        ax01.semilogy(polynom_list[2:], abs(err_list[2:]), 'k')
        ax011.semilogy(polynom_list[3:], abs(np.diff(err_list))[2:], 'b', )
        ax011.semilogy(polynom_list[3:], abs(np.diff(err_list))[2:], '.b')
        ax011.semilogy([polynom_list[1], polynom_list[-1]], [eps, eps], '--b')
    elif len(err_list.shape) > 1:
        for err_list in err_list:
            ax01.semilogy(polynom_list[2:], abs(err_list[2:]), '.k')
            ax01.semilogy(polynom_list[2:], abs(err_list[2:]), 'k')
            ax011.semilogy(polynom_list[3:], abs(np.diff(err_list))[2:], 'b', )
            ax011.semilogy(polynom_list[3:], abs(np.diff(err_list))[2:], '.b')
    ax011.semilogy([polynom_list[1], polynom_list[-1]], [eps, eps], '--r', label='E={}'.format(np.round(eps,1)))
    ax011.legend()
    ax01.set_xlabel('Polynomial order')
    ax01.set_ylabel('Error, $|\delta |^2$')
    ax011.set_ylabel('$|\Delta \delta |^2$', color='blue')
    
    ax011.tick_params(axis='y', colors='blue')
    ax011.grid(axis='y', color='blue')
    
    myFmt = dates.DateFormatter('%H:%M')
    ax02.xaxis.set_major_formatter(myFmt)
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax00.get_xticklabels(), visible=False)
    ax02.xaxis.set_major_formatter(myFmt)
    
    plt.tight_layout()
    if odir is not None:
        if not os.path.exists(odir):
            import subprocess
            subprocess.call('mkdir -p {}'.format(odir), shell=True, timeout=2)
        sfn = 'rxi{}_svi{}.png'.format(irx, isv)
        plt.savefig(odir + sfn, dpi=100)
        plt.close(fig)
    elif PLOT is False:
        plt.close(fig)
    else:
        plt.show()
    
def tecPerLOS(D, maxgap=10, maxjump=1):
    C1 = D['C1'].values
    C1[~idel] = np.nan
    C2 = D['P2'].values
    C2[~idel] = np.nan
    L1 = D['L1'].values
    L1[~idel] = np.nan
    L2 = D['L2'].values
    L2[~idel] = np.nan
    
    stec = np.nan * np.ones(dt.shape[0])
    
    if np.sum(np.isfinite(C1)) < (15 / (60/tsps)): 
        # If shorter than 15 minutes, skip
        return stec, []
    
    ixin, intervals = pyGnss.getIntervals(L1,L2,C1,C2, maxgap=maxgap, maxjump=maxjump)
    
    for ir, r in enumerate(intervals):
        if r[-1] - r[0] < (15 / (60/tsps)):
            intervals.pop(ir)
        else:
            stec[r[0]:r[-1]] = pyGnss.slantTEC(C1[r[0]:r[-1]], C2[r[0]:r[-1]], 
                                          L1[r[0]:r[-1]], L2[r[0]:r[-1]])
    
    stec_zero_bias = np.nanmin(stec)
    stec -= stec_zero_bias - 5
    
    return stec, intervals

def tecdPerLOS(stec, intervals, mask, eps=1, polynom_list=None, 
               zero_mean=False, filter='polynomial'):
    tecd = np.nan * np.ones(stec.size)
            
    for ir, r in enumerate(intervals):
        chunk = stec[r[0]+1 : r[1]-1]
        idf = np.isfinite(chunk)
        if np.sum(np.isfinite(chunk)) < (15 * (60/tsps)): 
            continue
        if np.sum(np.isnan(chunk)) > 0:
            chunk = gu.cubicSplineFit(chunk, idf)
        
        if filter == 'polynomial':
            res, err_list0, po  = gu.detrend(chunk, polynom_list=polynom_list, eps=eps, mask=mask[r[0]+1 : r[1]-1], polynomial_order=True)
        polynom_orders.append([np.squeeze(np.diff(r)) / (60/tsps), po])
        delta_eps.append(abs(np.diff(err_list0)[-1]))
        if ir == 0 or 'err_list' not in locals():
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

PLOT = 1
#saveroot = 'C:\\Users\\smrak\\Documents\\data\\detrending\\232\\plots_v0\stec_dtec\\'
saveroot = '/media/smrak/gnss/test/plots_249/'
#saveroot = None

day = 249

#root = 'C:\\Users\\smrak\\Google Drive\\BU\\Projects\\PhD\\detrending\\'
#obsroot = root + str(day) + '\\rnx\\'
#fnav = root + str(day) + '\\supp\\brdc{}0.17n'.format(day)
#fjplg = root + str(day) + '\\supp\\jplg{}0.17i'.format(day)

root = '/media/smrak/gnss/test30/'
root = '/media/smrak/gnss/'
obsroot = root + 'obs/highrate/2017/{}/'.format(day)
obsfnlist = glob(obsroot + '*.nc')
fnav = root + '/nav/brdc{}0.17n'.format(day)
fsp3 = root + 'nav/igs{}0.17sp3'.format(day)
fjplg = root + '/jplg/jplg{}0.17i'.format(day)


# Satbias
#satbias = pyGnss.getSatBias(fjplg)
# Processing options
satpos = True
args = ['L1', 'L2']
tlim = None
el_mask = 30


polynom_list = np.arange(0,20)

isvnumber = None
irxnumber = 181

# Stats
polynom_orders = []
delta_eps = []
for irx, fnc in enumerate(obsfnlist):
    
    # Data
    svlist = gr.load(fnc).sv.values
    navdata = gr.load(fnav)
    if irxnumber is not None:
        fnc = obsfnlist[irxnumber]
#    if irx > 3:
#        break
    for isv, sv in enumerate(svlist):
        if isvnumber is not None:
            sv = svlist[isvnumber]
        try:
            el_mask_in = el_mask - 10 if (el_mask - 10) >= 8 else 8
            D = pyGnss.dataFromNC(fnc,fnav,sv=sv,fsp3=fsp3,tlim=tlim,el_mask=el_mask_in, satpos=satpos)
            # Remove inital recovery at time 00:00
            mask0 = np.ones(D.time.values.size, dtype=bool)
            mask0[:3] = False
            
            # Merge with the elevation mask
            idel = np.logical_and(D['idel'].values, mask0)
#            sb = satbias[sv]
            
            dt = D.time.values
            tsps = np.diff(dt.astype('datetime64[s]'))[0].astype(int)
            eps = 1 * np.sqrt(30/tsps)
            elv = D.el.values
            
            mask = (np.nan_to_num(elv) >= el_mask)
            
            stec, intervals = tecPerLOS(D, maxjump=1, maxgap=10)
            F = np.nan * np.copy(stec)
            F[np.isfinite(elv)] = pyGnss.getMappingFunction(elv[np.isfinite(elv)], 350)
            stec *= F
            if np.sum(np.isfinite(stec)) < (15 * (60/tsps)): 
                # If shorter than 15 minutes, skip
                continue
            tecd, err_list = tecdPerLOS(stec, intervals, mask, polynom_list=polynom_list, eps=eps)
            # Print the shit
            tecd[~mask] = np.nan
            stec[~mask] = np.nan
            elv[np.isnan(tecd)] = np.nan
            if PLOT:
                plots(dt, stec, elv, tecd, polynom_list, err_list, odir=None)
        except BaseException as e:
            print (e)
        if isvnumber is not None:
            break
    if irxnumber is not None:
        break
    

#D = h5py.File(saveroot+'diagnostic.h5', 'w')
#po_length = np.array(polynom_orders)[:,0]
#po = np.array(polynom_orders)[:,1]
##D.create_dataset('po', data=po)
##D.create_dataset('po_length', data=po_length)
##D.close()
#
#fig = plt.figure(figsize=[8,5])
#plt.title('Pool size: {}'.format(po.size))
#plt.hist(list(po), range=(0,20), bins=20, density=True, color='b', align='left', rwidth=0.9)
#plt.xlabel('Polynomial order')
#plt.ylabel('Probability')
#plt.xlim([3,polynom_list[-1]])
##plt.savefig(saveroot + 'po_hist.png', dpi=100)
##plt.close(fig)
#
#fig = plt.figure(figsize=[8,5])
#plt.title('Pool size: {}'.format(po.size))
#plt.hist(list(po_length), bins=20, density=True, color='b', align='left', rwidth=0.9)
#plt.xlabel('LOS length [min]')
#plt.ylabel('Probability')
##plt.savefig(saveroot + 'polength_hist.png', dpi=100)
##plt.close(fig)
#
#fig = plt.figure(figsize=[8,5])
#h, x0, y0, im = plt.hist2d(po, po_length, bins=[16,50], range=[[4,20], [0,500]])
#plt.xlabel('Polynomial order')
#plt.ylabel('LOS length [min]')
#plt.colorbar()
#plt.clim([0, int(0.75 * np.nanmax(h))])
##plt.savefig(saveroot + 'po_length_hist2d.png', dpi=200)
##plt.close(fig)
