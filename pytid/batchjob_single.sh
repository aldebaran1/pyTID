#!/bin/bash -l

if [ $1 =  "scint" ]; then
 cfg=scint_processing.yaml
 FLAGS=("--ts 1 --elmask 30 --log --stec --cfg scint_processing.yaml")
else
 echo Default settings are set for tid processing.
 cfg=tid_processing.yaml
 FLAGS=("--ts 60 --elmask 20 --log --cfg $cfg")
fi

if [ $2 ]; then
 date=$2
else
 echo Missing date
 exit 1 
fi

if [ $3 ]; then
 file=$3
else
 echo Missing file name 
 exit 1
fi 

python nc2tec_v1.py $date $file $FLAGS
