#!/usr/bin/env bash

#$ -l mem_total=100
#$ -pe omp 4
#$ -l h_rt=30:00:00
#$ -N s2ix  

python scint2ix.py $1 --log 
