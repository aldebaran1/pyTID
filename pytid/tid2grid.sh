#!/bin/bash -l
#-N tid2grid
#-l h_rt 48:00:00

dir=/projectnb/semetergrp/tid/hdfgrid/

flags=("--ofn $dir --mode aer -r 0.3 -x -140 -55 -y 0 60")

python ~/pyTID/pytid/tid2grid_v2.py $1 $flags --altkm 250
python ~/pyTID/pytid/tid2grid_v2.py $1 $flags --altkm 350
python ~/pyTID/pytid/tid2grid_v2.py $1 $flags --altkm 450
python ~/pyTID/pytid/tid2grid_v2.py $1 $flags --altkm 550
