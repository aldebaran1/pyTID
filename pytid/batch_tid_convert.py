#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:01:28 2019

@author: smrak
"""

import subprocess
from datetime import datetime, timedelta
import numpy as np
from glob import glob
import os

fdir = '/media/smrak/gnss/obs/2018/'
start = '0201'
stop = '0210'

FLAGS = "--elmask 20 --ts 60 --log"

dir_list = np.array(os.listdir(fdir))
len_routine = np.vectorize(len)
dir_ix = (len_routine(dir_list) == 4)
dirs = dir_list[dir_ix]

# Iterte thru dates
path = os.path.normpath(fdir)
try:
    year = int(path.split(os.sep)[-1])
except:
    year = 2017

day0 = datetime.strptime("{} {}".format(year, start), "%Y %m%d")
day1 = datetime.strptime("{} {}".format(year, stop), "%Y %m%d")
d = day0
day_list = []
while d <= day1:
    mm = str(d.month) if len(str(d.month)) == 2 else '0' + str(d.month)
    dd = str(d.day) if len(str(d.day)) == 2 else '0' + str(d.day)
    day_list.append(mm + dd)
    d += timedelta(days=1)

for d in day_list:
    if os.path.exists(os.path.join(fdir, d)):
        date = "{}-{}-{}".format(year, int(d[:2]), int(d[-2:]))
        rxlist = fdir + 'conus' + d + '.yaml'
        assert os.path.exists(rxlist), "yaml file does not exist"
        print ("Converting {}".format(date))
        subprocess.call("nohup python nc2tec_v1.py {} {} {} > batch_convet_log.log".format(date, rxlist, FLAGS), shell=True)
    else:
        print ("Dolder with data for {} does not exist.".format(d))
