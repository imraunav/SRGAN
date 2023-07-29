#!/bin/bash

#job name
#PBS -N SRResNet
#resource requested
#PBS -l nodes=1:ppn=8
#name of queue
#PBS -q external
#output and error file
#PBS -o output-srresnet-improve.o
#PBS -e error-srresnet-improve.e
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

python train-srresnet-improve.py

echo "End time:"
date