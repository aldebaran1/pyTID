#!/bin/bash -l
dir=/projectnb/semetergrp/tid/images/
flags=("--odir $dir --cfg cfg/tid.yaml -m map/conus.yaml")
python gps2d.py $1 $flags
