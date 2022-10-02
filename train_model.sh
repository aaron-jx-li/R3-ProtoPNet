#!/bin/bash


#SBATCH --nodes 1
#SBATCH --job-name=trainProtoPNet
#SBATCH --cpus-per-task 4


python main.py