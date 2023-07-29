#!/bin/bash

#job name
#PBS -N SRGAN
#resource requested
#PBS -l nodes=1:ppn=16
#name of queue
#PBS -q external
#output and error file
#PBS -o output-srgan.o
#PBS -e error-srgan.e
#specify time required for job completion

#Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $t22104@students.iitmandi.ac.in

cd $PBS_O_WORKDIR
echo "Working directory: "
echo $PBS_O_WORKDIR
echo "Running on: " 
echo $PBS_O_HOST
echo "Start time: "
date
echo "Which python used: " 

source $HOME/anaconda3/bin/activate raunav-tf
which python

echo "Program Output begins: "

python train-srgan.py

echo "End time:"
date