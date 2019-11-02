#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:51:07 2019

@author: smrak
"""

from glob import glob
import os
import numpy as np
import subprocess
from dateutil import parser
from datetime import datetime
from argparse import ArgumentParser

###
def main(folder='', keyword='*.h5', treq=None, 
         altkm='350', resolution=0.3, timeout=60, xlim=None, ylim=None):
    altkm = np.array(altkm.split(','), dtype=float)
    if treq is not None:
        treq = treq.split(',')
        if len(treq) == 2:
            treqdt = [parser.parse(treq[0]), parser.parse(treq[1])]
        elif len(treq) == 1:
            treqdt = [datetime(1970, 1, 1), parser.parse(treq[0])]
        else:
            print ('Wrong "treq" value.')
            exit()
        
    ###
    filelist = sorted(glob(folder + keyword))
    for f in filelist:
        fn = os.path.split(f)[1]
        for H in altkm:
            try:
                FLAGS = "--mode aer --altkm {} -r {} -x {} {} -y {} {}".format(int(H), resolution, xlim[0], xlim[1], ylim[0], ylim[1])
            except BaseException as e:
                print (e)
        
            date = '{}-{}-{}'.format(fn[:4], fn[5:7], fn[7:9])
            try:
                datedt = parser.parse(date)
                if treq is None:
                    cmd = "nohup python tid2grid_v2.py {} {} > {}.log".format(f, FLAGS, date.replace('-',''))
                    print ("Processing {} ... FLAGS: {}".format(date, FLAGS))
                    subprocess.call(cmd, shell=True, timeout=timeout*60)
                else:
                    if datedt >= treqdt[0] and datedt <= treqdt[1]:
                        cmd = "nohup python tid2grid_v2.py {} {} > {}.log".format(f, FLAGS, date.replace('-',''))
                        print ("Processing {} ... FLAGS: {}".format(date, FLAGS))
                        subprocess.call(cmd, shell=True, timeout=timeout*60)
            except:
                pass

if __name__ == '__main__':
    
    p = ArgumentParser()
    p.add_argument('directory', type = str, help = 'path to .hdf5 files. Example: /media/name/gps/hdf/')
    p.add_argument('keyword', type = str, help = 'glob search thru the directiry, default=*.h5', default='*.h5')
    p.add_argument('--treq', type = str, help='start,en for start date for processing. deafult is None. "2018-8-20,2018-8-23"<- comma separated. Or Just end date "2018-8-15".', default = None)
    p.add_argument('--altkm', type = str, help = "altkm, single of a sequnce, comma separated. '150, 250, 350'", default = '350')
    p.add_argument('-r', '--resolution', type=float, help = 'Grid resolution. Default=0.3.', default = 0.3)
    p.add_argument('-x', help = 'Longitude limits for the grid. Defult=-135, -55', default = [-135, -55])
    p.add_argument('-y', help = 'Latitude limits for the grid. Defult=5, 52', nargs=2, default = [2, 52])
    p.add_argument('--timeout', help = 'timeout for processing task? Default=60min?', default = 60)
    P = p.parse_args()
    
    main(folder=P.directory, keyword=P.keyword, treq=P.treq, 
         altkm=P.altkm, resolution=P.resolution, timeout=P.timeout, 
         xlim=P.x, ylim=P.y)